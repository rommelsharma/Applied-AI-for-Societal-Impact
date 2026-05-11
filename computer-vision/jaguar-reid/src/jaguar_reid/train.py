from __future__ import annotations

import torch
import os

def print_gpu_info():
    print(f"PyTorch version: {torch.__version__}")
    # Check for CUDA (NVIDIA GPU) availability
    if torch.cuda.is_available():
        print("GPU Available. Details below:")
        print('PyTorch:', torch.__version__,  '\nCUDA   :', torch.version.cuda, '\nArch   :', torch.cuda.get_arch_list())
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"Current CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is NOT available. PyTorch will use your CPU.")

"""
Jaguar Re-ID v6 GPU — High-Performance Pipeline
================================================
Upgraded from v4 with three evidence-based improvements:

NEW in v6
---------
* GeM Pooling (learnable p): replaces GlobalAvgPool. Backbone now calls
  forward_features() so GeM receives the full spatial feature map.
  Proven +1-3% mAP on fine-grained retrieval benchmarks.
* LLRD (Layer-wise LR Decay): backbone layers trained at progressively
  lower LRs (decay=0.8/layer from output). Prevents catastrophic forgetting
  of MegaDescriptor's wildlife-specific features; head trained 10× faster.
* Query Expansion (AQE, k=5): each embedding is replaced by the mean of
  itself + 5 nearest neighbours before re-ranking. Smooths single-image
  noise. Applied before k-reciprocal re-ranking for compounding gain.

These were selected specifically for real-world robustness, not overfitting
to a leaderboard. All changes preserve full cross-validation and per-epoch
mAP monitoring. No submission blending or leakage.

Key upgrades over v3
--------------------
* EPOCHS 12 → 40   : DINOv2 needs more steps on a small dataset (1896 imgs)
* FOLDS  3  → 5    : 80% train / 20% val per fold; more training data
* LR 2e-4 → 5e-5   : lower LR avoids destabilising pre-trained ViT weights
* BATCH 8  → 16    : doubled batch size with GRAD_ACCUM 4→2 keeps effective=32
* PK_IDENTITIES = 4   # 4 × 4 = 16 = BATCH_SIZE
* ArcFace s=30→64, m=0.35→0.50 : standard high-precision re-ID values
* EMBEDDING_DIM 512→768 : matches ViT-B/14 feature dim; removes projection bottleneck
* LABEL_SMOOTHING 0.05→0.10, TRIPLET_WEIGHT 0.15→0.30 : more regularisation
* USE_ALPHA True : alpha mask removes background, focuses model on jaguar body
* VAL_EVERY_EPOCHS 2→1 : catch peak mAP epoch in 40-epoch runs
* 4x TTA at inference : avg of orig + hflip + vflip + both → ~+2-4% mAP
* k-reciprocal re-ranking (Zhong et al. 2017) at inference → +5-15% mAP
* IMPORTANT: DELETE output/cache/ before running if previous run was on MPS
  with the broken backward pass (stale phase markers skip re-training)

Run on GCP/Kaggle (CUDA) for best results. MPS is supported for local testing.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

try:
    import timm
except ImportError as exc:
    raise ImportError("timm is required. Install with `pip install timm`.") from exc

# ══════════════════════════════════════════════════════════════════════════════
# GPU MEMORY MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def free_gpu_memory(*refs) -> None:
    for r in refs:
        try:
            del r
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

def gpu_mem_report() -> str:
    if not torch.cuda.is_available():
        return "no-cuda"
    alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    res   = torch.cuda.memory_reserved()   / (1024 ** 3)
    return f"GPU alloc={alloc:.2f} GB reserved={res:.2f} GB"


# ====================================================
# HuggingFace Connect for read access
# =====================================================

def hf_connect():
    # Load HF token from Kaggle Secrets and authenticate
    try:
        # from kaggle_secrets import UserSecretsClient
        # secrets = UserSecretsClient()
        # hf_token = secrets.get_secret("HF_TOKEN")
        # os.environ["HF_TOKEN"] = hf_token
        hf_token = os.environ["HF_TOKEN"]
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print("✓ HuggingFace authenticated successfully")
    except Exception as e:
        print(f"⚠ HF auth skipped (running outside Kaggle or secret missing): {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MPS COMPATIBILITY PATCHES
# ══════════════════════════════════════════════════════════════════════════════

def patch_timm_attention_for_mps(model: nn.Module, device: torch.device) -> None:
    """Two-part MPS fix for timm ViT models.

    SOURCE 1 — PatchEmbed output non-contiguous: forward hook forces .contiguous().
    SOURCE 2 — Attention.forward backward .view crash: replace .view with .reshape.
    """
    if device.type != "mps":
        return

    import types

    backbone = getattr(model, "backbone", model)
    if hasattr(backbone, "patch_embed"):
        def _pe_hook(module, inp, out):
            if isinstance(out, torch.Tensor) and not out.is_contiguous():
                return out.contiguous()
        backbone.patch_embed.register_forward_hook(_pe_hook)
        print("  MPS: PatchEmbed contiguous hook registered.")

    try:
        from timm.layers.attention import Attention as TimmAttn
    except ImportError:
        print("  MPS patch: timm.layers.attention not found — skipping.")
        return

    def _mps_safe_forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        # .reshape instead of .view — tolerates non-contiguous gradients on MPS backward
        x = x.transpose(1, 2).contiguous().reshape(B, N, self.attn_dim)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    n = 0
    for module in model.modules():
        if type(module) is TimmAttn:
            module.forward = types.MethodType(_mps_safe_forward, module)
            n += 1
    if n:
        print(f"  MPS: patched {n} Attention blocks (.reshape in forward).")
    else:
        print("  MPS: no timm Attention blocks found to patch.")

# ✅ Fixed — skips compile on Windows where Triton is unavailable
def try_compile_model(model: nn.Module, device: torch.device) -> nn.Module:
    if device.type != "cuda":
        return model
    # torch.compile requires Triton which has no Windows support
    if sys.platform == "win32":
        print("  torch.compile skipped (Windows — Triton unavailable).")
        return model
    try:
        cap_major, _ = torch.cuda.get_device_capability(device)
        if cap_major < 7:
            print(f"  torch.compile skipped (GPU SM{cap_major}x < SM70; Triton requires SM70).")
            return model
    except Exception:
        pass
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile enabled (reduce-overhead mode).")
        return compiled
    except Exception as exc:
        print(f"  torch.compile unavailable ({exc}); running eagerly.")
        return model

def enable_gradient_checkpointing(model: nn.Module, device: torch.device) -> None:
    
    if device.type == "mps":
        print("  Gradient checkpointing skipped on MPS (unsupported; use CUDA only).")
        return
    
    backbone = getattr(model, "backbone", model)
    
    if hasattr(backbone, "set_grad_checkpointing"):
        backbone.set_grad_checkpointing(enable=True)
        print("  Gradient checkpointing enabled on backbone.")
    else:
        print("  Backbone does not support set_grad_checkpointing; skipping.")

# ══════════════════════════════════════════════════════════════════════════════
# GCP HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)

def install_dependencies() -> None:
    run_cmd([sys.executable, "-m", "pip", "install", "-U", "pip",
             "torch", "torchvision", "timm", "pandas", "numpy",
             "scikit-learn", "matplotlib", "Pillow", "tqdm"])

def gcs_pull(gcs_uri: str, local_dest: Path) -> None:
    if local_dest.exists():
        shutil.rmtree(local_dest)
    local_dest.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(["gsutil", "-m", "cp", "-r", gcs_uri, str(local_dest.parent)])

def gcs_push(local_src: Path, gcs_uri: str) -> None:
    run_cmd(["gsutil", "-m", "cp", "-r", str(local_src), gcs_uri])

# ══════════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY & DEVICE
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def resolve_num_workers(requested: int, device: torch.device) -> int:
    if device.type in ("mps", "cpu"):
        return min(requested, 2)
    cpu_cores = os.cpu_count() or 2
    auto = max(2, min(cpu_cores // 2, 8))
    resolved = min(requested, auto) if requested > 0 else auto
    if resolved != requested:
        print(f"  NUM_WORKERS auto-scaled: {requested} → {resolved}")
    return resolved

def select_best_device() -> torch.device:
    # TPU/XLA check first
    try:
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device()
        print(f"Selected TPU/XLA device: {dev}")
        return dev
    except ImportError:
        pass
    if torch.cuda.is_available():
        best_idx, best_score = 0, -1.0
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            score = (props.total_memory / (1024 ** 3)) * max(1, getattr(props, "multi_processor_count", 1))
            if score > best_score:
                best_score, best_idx = score, i
        torch.cuda.set_device(best_idx)
        props = torch.cuda.get_device_properties(best_idx)
        print(f"Selected CUDA:{best_idx} — {props.name} | VRAM={props.total_memory/(1024**3):.2f} GB")
        return torch.device(f"cuda:{best_idx}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Selected MPS device (Apple Metal).")
        return torch.device("mps")
    print("No GPU found. Falling back to CPU.")
    return torch.device("cpu")

# ══════════════════════════════════════════════════════════════════════════════
# CACHING & PHASE MARKERS
# ══════════════════════════════════════════════════════════════════════════════

def _json_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]

def file_signature(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}

def png_dir_signature(root: Path) -> dict:
    files = sorted(root.rglob("*.png"))
    if not files:
        return {"path": str(root.resolve()), "count": 0, "total_size": 0, "latest_mtime_ns": 0}
    total_size, latest = 0, 0
    for p in files:
        st = p.stat()
        total_size += int(st.st_size)
        latest = max(latest, int(st.st_mtime_ns))
    return {"path": str(root.resolve()), "count": len(files),
            "total_size": int(total_size), "latest_mtime_ns": int(latest)}

def cache_dir(output_dir: Path) -> Path:
    d = output_dir / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d

def load_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def save_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))

def phase_marker_path(output_dir: Path, phase_name: str) -> Path:
    return cache_dir(output_dir) / f"phase_{phase_name}_done.json"

def is_phase_complete(output_dir: Path, phase_name: str) -> bool:
    return phase_marker_path(output_dir, phase_name).exists()

def mark_phase_complete(output_dir: Path, phase_name: str, meta: dict | None = None) -> None:
    payload = {"phase": phase_name, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if meta:
        payload.update(meta)
    save_manifest(phase_marker_path(output_dir, phase_name), payload)
    print(f"  Phase '{phase_name}' complete ✓")

def clear_phase(output_dir: Path, phase_name: str) -> None:
    p = phase_marker_path(output_dir, phase_name)
    if p.exists():
        p.unlink()

# ══════════════════════════════════════════════════════════════════════════════
# AMP HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def create_grad_scaler(device: torch.device, enabled: bool):
    class _NoOpScaler:
        @staticmethod
        def scale(loss): return loss
        @staticmethod
        def step(optimizer): optimizer.step()
        @staticmethod
        def update(): return None
    if not enabled or device.type != "cuda":
        return _NoOpScaler()
    try:
        return torch.amp.GradScaler(device="cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)

def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type=device.type, enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=(enabled and device.type == "cuda"))

# ══════════════════════════════════════════════════════════════════════════════
# IMAGE SIZE / PATH RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def infer_required_img_size(backbone: nn.Module) -> int | None:
    patch_embed = getattr(backbone, "patch_embed", None)
    if patch_embed is not None and hasattr(patch_embed, "img_size"):
        img_size = getattr(patch_embed, "img_size")
        if isinstance(img_size, (tuple, list)) and len(img_size) >= 1:
            return int(img_size[0])
        if isinstance(img_size, int):
            return int(img_size)
    default_cfg = getattr(backbone, "default_cfg", None) or {}
    in_size = default_cfg.get("input_size", None)
    if isinstance(in_size, (tuple, list)) and len(in_size) == 3:
        return int(in_size[1])
    return None

def resolve_effective_img_size(backbone: nn.Module, requested_img_size: int) -> int:
    required = infer_required_img_size(backbone)
    if required is not None and requested_img_size != required:
        print(f"  Backbone requires img_size={required} (requested {requested_img_size}). Using {required}.")
        return required
    return requested_img_size

def resolve_image_root(base_dir: Path, split: str) -> Path:
    candidates = [
        base_dir / split,
        base_dir / split / split,
        base_dir / "images" / split,
        base_dir / f"{split}_images",
    ]
    for c in candidates:
        if c.exists() and any(c.rglob("*.png")):
            return c
    raise FileNotFoundError(f"No PNG images found for split={split} under {base_dir}")

# ══════════════════════════════════════════════════════════════════════════════
# DATASET & TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

def dhash(image: Image.Image, hash_size: int = 8) -> str:
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = "".join("1" if b else "0" for b in diff.flatten())
    return hex(int(bits, 2))[2:].rjust(hash_size * hash_size // 4, "0")

def make_train_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.8),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.0), value="random"),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),  # dynamic for DINOv2/MegaDescriptor
    ])

# ✅ Fixed — uses the same NORM_MEAN/NORM_STD as training
def make_eval_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

@dataclass
class FoldIndices:
    train_idx: np.ndarray
    val_idx:   np.ndarray

def build_stratified_image_folds(df: pd.DataFrame, n_splits: int, seed: int) -> List[FoldIndices]:
    y = df["ground_truth"].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [FoldIndices(train_idx=tr, val_idx=va) for tr, va in skf.split(df.index.values, y)]

class JaguarDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_root: Path,
        transform: transforms.Compose,
        label_to_idx: Dict[str, int] | None = None,
        use_alpha: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform  = transform
        self.use_alpha  = use_alpha
        self.label_to_idx = label_to_idx
        self._direct_paths   = {fn: (self.image_root / fn) for fn in self.df["filename"].tolist()}
        self._resolved_paths: Dict[str, Path] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, filename: str) -> Image.Image:
        p = self._resolved_paths.get(filename)
        if p is None:
            p = self._direct_paths.get(filename, self.image_root / filename)
            if not p.exists():
                matches = list(self.image_root.rglob(filename))
                if not matches:
                    raise FileNotFoundError(f"Cannot resolve image: {filename}")
                p = matches[0]
            self._resolved_paths[filename] = p
        with Image.open(p) as im:
            img = im.convert("RGBA")
        if self.use_alpha:
            rgba  = np.asarray(img, dtype=np.float32) / 255.0
            alpha = rgba[..., 3:4]
            rgb   = rgba[..., :3] * alpha
            img   = Image.fromarray((rgb * 255.0).astype(np.uint8))  # mode inferred from shape
        else:
            img = img.convert("RGB")
        return img

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = self._load_image(row["filename"])
        x   = self.transform(img)
        if "ground_truth" in self.df.columns and self.label_to_idx is not None:
            y = self.label_to_idx[row["ground_truth"]]
            return x, y, row["filename"]
        return x, row["filename"]

# ══════════════════════════════════════════════════════════════════════════════
# SAMPLER
# ══════════════════════════════════════════════════════════════════════════════

class PKSampler(Sampler[int]):
    """P identities × K instances per mini-batch."""

    def __init__(self, labels: Sequence[int], p: int, k: int, length: int, seed: int = 42):
        self.labels = np.array(labels)
        self.p, self.k, self.length = p, k, length
        self.rng = np.random.default_rng(seed)
        self.label_to_indices: Dict[int, np.ndarray] = {
            int(lab): np.where(self.labels == lab)[0]
            for lab in np.unique(self.labels)
        }
        self.unique_labels = np.array(sorted(self.label_to_indices.keys()))

    def __iter__(self):
        n_batches = math.ceil(self.length / (self.p * self.k))
        out: List[int] = []
        for _ in range(n_batches):
            labs = self.rng.choice(self.unique_labels, size=min(self.p, len(self.unique_labels)), replace=False)
            for lab in labs:
                idxs   = self.label_to_indices[int(lab)]
                chosen = self.rng.choice(idxs, size=self.k, replace=len(idxs) < self.k)
                out.extend(chosen.tolist())
        return iter(out[: self.length])

    def __len__(self) -> int:
        return self.length

# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# GEM POOLING
# Generalised Mean Pooling with learnable exponent p.
# Replaces GlobalAvgPool after backbone: emphasises peak activations rather
# than averaging them. Proven gain in fine-grained retrieval (+1-3% mAP).
# p is initialised to 3.0 and learned during training.
# Reference: Radenovic et al. 2019 "Fine-tuning CNN Image Retrieval..."
# ══════════════════════════════════════════════════════════════════════════════

class GeM(nn.Module):
    """Generalised Mean Pooling with trainable exponent p."""
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x  # (B, C) already pooled — skip

        if x.dim() == 3:
            # (B, N, C) patch tokens — reshape to (B, C, H, W) then pool
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            if H * W == N:
                x = x.permute(0, 2, 1).reshape(B, C, H, W)
            else:
                # CLS token present: drop it and pool the rest
                x = x[:, 1:, :].permute(0, 2, 1)
                N2 = x.shape[2]
                H = W = int(math.sqrt(N2))
                x = x.reshape(B, C, H, W)

        elif x.dim() == 4:
            # Swin outputs channels-last (B, H, W, C) where C >> H, W
            # CNNs output channels-first (B, C, H, W) where C >> H, W too —
            # but for Swin at 384px: (B, 12, 12, 1536) → shape[-1]=1536 > shape[1]=12
            if x.shape[-1] > x.shape[1]:  # channels-last detected
                x = x.permute(0, 3, 1, 2).contiguous()  # → (B, C, H, W)

        # x is now guaranteed (B, C, H, W)
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p).flatten(1)

class ArcMarginProduct(nn.Module):
    # s=64, m=0.50 are the standard high-precision re-ID values.
    # s=30/m=0.35 (v3 defaults) under-utilise angular space with only 31 classes.
    def __init__(self, in_features: int, out_features: int, s: float = 64.0, m: float = 0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = s, m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(emb), F.normalize(self.weight))
        sine   = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))
        phi    = cosine * self.cos_m - sine * self.sin_m
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.reshape(-1, 1), 1.0)
        return ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.s


class JaguarReIDModel(nn.Module):
    """
    Backbone → GeM Pool → BN → Linear → L2-norm → ArcFace
    
    Changes vs v4
    -------------
    * GeM pooling replaces GlobalAvgPool: learns per-channel emphasis.
      Especially helpful for Swin/ViT hybrids where spatial structure matters.
    * BatchNorm sits between pool and linear (standard re-ID practice).
    * forward_features() is called explicitly so GeM receives the spatial map,
      not the already-pooled scalar from timms default head.
    """

    def __init__(self, backbone_name, num_classes, embedding_dim=768,
                 img_size=384, gem_p=3.0):
        super().__init__()
        self.backbone = self._build_backbone(backbone_name, img_size)
        
        # ── Probe actual feature dim with a dummy forward pass ──
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feat_out = self.backbone.forward_features(dummy)
            # GeM will be applied next — simulate it to get the final shape
            if feat_out.dim() == 4:
                # Use max() — channel dim is always the largest value
                # Works for: Swin (B,H,W,C), CNN (B,C,H,W), square or non-square spatial
                feat_dim = max(feat_out.shape[1:])
            elif feat_out.dim() == 3:  # (B, N, C) patch tokens
                feat_dim = feat_out.shape[2]
            else:  # (B, C) already pooled
                feat_dim = feat_out.shape[1]
        
        print(f"  Probed feature dim: {feat_dim}")
        self.gem = GeM(p=gem_p)
        self.bn = nn.BatchNorm1d(feat_dim)          # ← correct dim now
        self.embedding = nn.Linear(feat_dim, embedding_dim)
        self.arc = ArcMarginProduct(embedding_dim, num_classes)


    @staticmethod
    def _build_backbone(backbone_name: str, img_size: int = 518) -> nn.Module:
        kwargs: dict = {
            "pretrained": True,
            "num_classes": 0,
            "img_size": img_size,
            "dynamic_img_size": True,
        }
        try:
            return timm.create_model(backbone_name, **kwargs)
        except TypeError:
            kwargs.pop("dynamic_img_size", None)
            return timm.create_model(backbone_name, **kwargs)
        except RuntimeError as exc:
            msg = str(exc)
            if "fc_norm" in msg and "norm." in msg:
                return timm.create_model(backbone_name, global_pool="", **kwargs)
            raise

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        # Use forward_features so GeM receives the spatial feature map,
        # not the backbone's pooled scalar output.
        feat = self.backbone.forward_features(x)
        feat = self.gem(feat)
        feat = self.bn(feat)
        emb  = F.normalize(self.embedding(feat), p=2, dim=1)
        if y is None:
            return emb
        return emb, self.arc(emb, y)

# ══════════════════════════════════════════════════════════════════════════════
# EARLY STOPPING
# ══════════════════════════════════════════════════════════════════════════════
# Monitors validation mAP each evaluated epoch. Stops training when no
# improvement > min_delta is seen for `patience` consecutive evaluated epochs.
# warmup_epochs: early stopping is disabled for the first N epochs so the
# model has time to escape the random-init loss plateau.
# On early stop, best weights are automatically restored from best_ckpt.
# All decisions are logged to cache/early_stop_fold{n}.json for auditability.

class EarlyStoppingMonitor:
    """Track validation mAP and signal when training should stop."""

    def __init__(
        self,
        patience: int = 8,
        min_delta: float = 1e-4,
        warmup_epochs: int = 10,
        fold_id: int = 0,
        output_dir: Path = Path("output"),
    ):
        self.patience       = patience
        self.min_delta      = min_delta
        self.warmup_epochs  = warmup_epochs
        self.fold_id        = fold_id
        self.output_dir     = output_dir
        self.best_map       = -1.0
        self.best_epoch     = 0
        self.wait           = 0          # consecutive non-improving eval epochs
        self.stopped_epoch  = 0
        self.should_stop    = False

    def step(self, epoch: int, val_map: float) -> bool:
        """Call after each validation. Returns True if training should stop."""
        if epoch <= self.warmup_epochs:
            # Still in warmup — update best silently but never trigger stop
            if val_map > self.best_map:
                self.best_map   = val_map
                self.best_epoch = epoch
            return False

        if val_map > self.best_map + self.min_delta:
            self.best_map   = val_map
            self.best_epoch = epoch
            self.wait       = 0
        else:
            self.wait += 1
            print(f"    EarlyStopping: no improvement for {self.wait}/{self.patience} "
                  f"eval epochs (best mAP={self.best_map:.5f} @ epoch {self.best_epoch})")

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.should_stop   = True
            print(f"  *** Early stopping triggered at epoch {epoch}. "
                  f"Best mAP={self.best_map:.5f} at epoch {self.best_epoch}. ***")
            self._save_log(epoch, "patience_exceeded")
            return True
        return False

    def _save_log(self, epoch: int, reason: str) -> None:
        log = {
            "fold_id":     self.fold_id,
            "stopped_epoch": epoch,
            "best_epoch":  self.best_epoch,
            "best_map":    self.best_map,
            "reason":      reason,
            "patience":    self.patience,
            "min_delta":   self.min_delta,
            "warmup_epochs": self.warmup_epochs,
            "saved_at":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        p = cache_dir(self.output_dir) / f"early_stop_fold{self.fold_id}.json"
        p.write_text(json.dumps(log, indent=2))
        print(f"  EarlyStopping log → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# LOSSES & METRICS
# ══════════════════════════════════════════════════════════════════════════════

class FocalCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets,
            weight=self.class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        probs = F.softmax(logits, dim=1)
        # .contiguous() prevents MPS backward .view crash on non-contiguous gather output
        pt    = probs.gather(1, targets.reshape(-1, 1)).contiguous().squeeze(1).clamp(1e-6, 1.0 - 1e-6)
        focal = (1.0 - pt).pow(self.gamma)
        return (focal * ce).mean()

def compute_class_weights(
    labels: Sequence[int], num_classes: int, mode: str = "effective_num", beta: float = 0.999,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    if mode == "inverse":
        w = 1.0 / counts
    elif mode == "effective_num":
        effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        w = 1.0 / np.clip(effective_num, 1e-8, None)
    else:
        raise ValueError(f"Unknown class weighting mode: {mode}")
    w = w / np.mean(w)
    return torch.tensor(w, dtype=torch.float32)

def batch_hard_triplet_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.2,
) -> torch.Tensor:
    embeddings = embeddings.contiguous()  # MPS: prevent non-contiguous grad in backward
    sq_norms   = (embeddings * embeddings).sum(dim=1)
    dot        = embeddings @ embeddings.t()
    dist       = torch.sqrt((sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2.0 * dot).clamp(min=1e-12))
    labels_col = labels.reshape(-1, 1)
    mask_same  = labels_col.eq(labels_col.t())
    eye        = torch.eye(dist.size(0), device=dist.device, dtype=torch.bool)
    hardest_pos = (dist * (mask_same & ~eye).float()).max(dim=1).values
    hardest_neg = (dist + mask_same.float() * 1e6).min(dim=1).values
    return F.relu(hardest_pos - hardest_neg + margin).mean()

def identity_balanced_map(emb: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
    sims = emb @ emb.T
    n = len(labels)
    id_to_aps: Dict[str, List[float]] = defaultdict(list)
    for i in range(n):
        rel = (labels == labels[i]).astype(np.int32)
        rel[i] = 0
        order = np.argsort(-sims[i])
        order = order[order != i]
        hits  = rel[order]
        n_rel = hits.sum()
        if n_rel == 0:
            continue
        cumsum = np.cumsum(hits)
        ranks  = np.arange(1, len(hits) + 1)
        precision_at_k = cumsum / ranks
        ap = (precision_at_k * hits).sum() / n_rel
        id_to_aps[str(labels[i])].append(float(ap))
    id_means  = {k: float(np.mean(v)) for k, v in id_to_aps.items() if v}
    macro_map = float(np.mean(list(id_means.values()))) if id_means else 0.0
    return macro_map, id_means


# ══════════════════════════════════════════════════════════════════════════════
# LAYER-WISE LEARNING RATE DECAY (LLRD)
# ══════════════════════════════════════════════════════════════════════════════
# Deep transformer layers capture generic features; shallower layers capture
# task-specific patterns. LLRD gives lower LR to deep (early) layers to
# prevent catastrophic forgetting, higher LR to shallow/head layers to adapt.
# layer_decay=0.8 means each layer group gets 80% of the LR of the group above.
# No_decay list prevents weight decay on biases and LayerNorm params (standard).

def build_llrd_optimizer(
    model: "JaguarReIDModel",
    lr: float,
    head_lr_multiplier: float = 10.0,
    layer_decay: float = 0.8,
    weight_decay: float = 0.05,
) -> torch.optim.AdamW:
    """AdamW with layer-wise LR decay for transformer backbones.

    The GeM pooling, BN, embedding linear, and ArcFace head are trained
    at lr × head_lr_multiplier (fast adaptation). Backbone layers are
    trained at progressively lower LRs from the output layer downward.
    """
    no_decay    = {"bias", "LayerNorm.weight", "LayerNorm.bias",
                   "norm.weight", "norm.bias", "bn.weight", "bn.bias"}
    head_names  = {"gem", "bn", "embedding", "arc"}
    param_groups: List[dict] = []

    # 1. Head params — high LR
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        top = name.split(".")[0]
        if top not in head_names:
            continue
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        param_groups.append({"params": [p], "lr": lr * head_lr_multiplier, "weight_decay": wd,
                              "_group_tag": f"head:{top}"})

    # 2. Backbone params — LLRD by block index
    backbone = model.backbone
    # Collect block containers (works for Swin, ViT, ConvNeXt, etc.)
    block_containers = []
    for attr in ("layers", "blocks", "stages"):
        ctr = getattr(backbone, attr, None)
        if ctr is not None and hasattr(ctr, "__len__") and len(ctr) > 0:
            if hasattr(ctr[0], "parameters"):
                block_containers = list(ctr)
            elif hasattr(ctr[0], "__iter__"):
                for sub in ctr:
                    block_containers.extend(list(sub))
            break

    n_layers = max(len(block_containers), 1)
    assigned: set = set()

    for layer_idx, block in enumerate(block_containers):
        # Closer to output = higher LR (smaller layer_decay exponent)
        depth_from_output = n_layers - layer_idx - 1
        block_lr = lr * (layer_decay ** depth_from_output)
        for name, p in block.named_parameters():
            if not p.requires_grad or id(p) in assigned:
                continue
            assigned.add(id(p))
            wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
            param_groups.append({"params": [p], "lr": block_lr, "weight_decay": wd,
                                  "_group_tag": f"backbone_layer{layer_idx}"})

    # 3. Any remaining backbone params (stem, patch_embed, etc.) at base LR
    for name, p in backbone.named_parameters():
        if not p.requires_grad or id(p) in assigned:
            continue
        assigned.add(id(p))
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        param_groups.append({"params": [p], "lr": lr, "weight_decay": wd,
                              "_group_tag": "backbone_stem"})

    n_params = sum(len(g["params"]) for g in param_groups)
    print(f"  LLRD: {len(param_groups)} param groups | {n_params} tensors | "
          f"head_lr={lr*head_lr_multiplier:.2e} | base_lr={lr:.2e} | decay={layer_decay}")
    return torch.optim.AdamW(param_groups)


# ══════════════════════════════════════════════════════════════════════════════
# K-RECIPROCAL RE-RANKING (Zhong et al. 2017)
# Typical gain: +5-15% mAP at inference with zero extra training.
# ══════════════════════════════════════════════════════════════════════════════

def k_reciprocal_rerank(emb: np.ndarray, k1: int = 20, k2: int = 6, lam: float = 0.3) -> np.ndarray:
    """Re-rank embedding matrix using k-reciprocal encoding."""
    n = emb.shape[0]
    S = (emb @ emb.T).astype(np.float32)
    V = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        order_i   = np.argsort(-S[i])
        nn_i      = order_i[1:k1 + 1]
        reciprocal = [j for j in nn_i if i in np.argsort(-S[j])[1:k1 + 1]]
        reciprocal_exp = list(reciprocal)
        for j in reciprocal:
            k1h   = int(round(k1 / 2))
            nn_j2 = np.argsort(-S[j])[1:k1h + 1]
            rr    = [x for x in nn_j2 if j in np.argsort(-S[x])[1:k1 + 1]]
            if len(rr) > (2 / 3) * len(reciprocal):
                reciprocal_exp.extend(rr)
        reciprocal_exp = list(set(reciprocal_exp))
        if reciprocal_exp:
            V[i, reciprocal_exp] = S[i, reciprocal_exp]
    # k2-NN query expansion on V
    V_qe = np.zeros_like(V)
    for i in range(n):
        top_k2 = np.argsort(-V[i])[:k2]
        rows   = np.append(top_k2, i)
        V_qe[i] = np.mean(V[rows], axis=0)
    orig_dist  = 1.0 - S
    jacc_dist  = 1.0 - V_qe
    final_sim  = 1.0 - ((1.0 - lam) * orig_dist + lam * jacc_dist)
    return 0.5 * (final_sim + final_sim.T)  # symmetrise

# ══════════════════════════════════════════════════════════════════════════════
# QUERY EXPANSION (AQE — Average Query Expansion)
# ══════════════════════════════════════════════════════════════════════════════
# After computing embeddings, replace each embedding with the average of itself
# and its top-k nearest neighbours in the embedding space.
# This smooths out noisy single-image embeddings using context from similar
# images — particularly effective on small datasets like this (1895 images).
# Typical gain: +1-4% mAP with zero training cost.
# Applied BEFORE k-reciprocal re-ranking.

def query_expansion(emb: np.ndarray, top_k: int = 5) -> np.ndarray:
    """Average Query Expansion: replace each embedding with mean of top-k neighbours."""
    print(f"  Applying Query Expansion (k={top_k})...")
    sims    = emb @ emb.T          # (N, N) cosine similarities (embeddings are L2-normed)
    indices = np.argsort(-sims, axis=1)[:, :top_k]
    new_emb = np.stack([emb[indices[i]].mean(axis=0) for i in range(len(emb))])
    norms   = np.linalg.norm(new_emb, axis=1, keepdims=True).clip(min=1e-8)
    return new_emb / norms         # re-normalise to unit sphere


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def fit_logistic_calibrator(scores: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    x   = torch.tensor(scores,  dtype=torch.float32).reshape(-1, 1)
    y   = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
    a   = nn.Parameter(torch.tensor([1.0]))
    b   = nn.Parameter(torch.tensor([0.0]))
    opt = torch.optim.LBFGS([a, b], max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x * a + b, y)
        loss.backward()
        return loss
    try:
        opt.step(closure)
        return float(a.detach()), float(b.detach())
    except Exception:
        return 1.0, 0.0

def apply_calibration(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    z   = np.clip(a * scores + b, -20.0, 20.0)
    out = 1.0 / (1.0 + np.exp(-z))
    if not np.isfinite(out).all():
        mn, mx = float(scores.min()), float(scores.max())
        out = (scores - mn) / (mx - mn) if mx > mn else np.full_like(scores, 0.5)
    return np.clip(out, 0.0, 1.0)

# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING EXTRACTION (chunked)
# ══════════════════════════════════════════════════════════════════════════════

def compute_embeddings(
    model: nn.Module, loader: DataLoader, device: torch.device, amp: bool,
) -> Tuple[np.ndarray, List[str], np.ndarray | None]:
    model.eval()
    all_emb, all_files, all_labels = [], [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Extract embeddings", leave=False):
            if len(batch) == 3:
                x, y, files = batch
                all_labels.append(y.numpy())
            else:
                x, files = batch
            x = x.to(device, non_blocking=True)
            with autocast_context(device=device, enabled=amp):
                emb = model(x, None)
            all_emb.append(emb.cpu().numpy())
            all_files.extend(list(files))
    emb_np    = np.vstack(all_emb) if all_emb else np.empty((0, 0), dtype=np.float32)
    labels_np = np.concatenate(all_labels) if all_labels else None
    return emb_np, all_files, labels_np


def compute_embeddings_chunked(
    model: nn.Module,
    dataset: JaguarDataset,
    device: torch.device,
    amp: bool,
    batch_size: int,
    num_workers: int,
    chunk_size: int,
    cache_prefix: Path,
) -> Tuple[np.ndarray, List[str]]:
    total   = len(dataset)
    n_chunks = math.ceil(total / chunk_size)
    all_emb_parts: List[np.ndarray] = []
    all_files:     List[str]        = []
    for ci in range(n_chunks):
        chunk_path = Path(f"{cache_prefix}_chunk{ci}.npz")
        start, end = ci * chunk_size, min((ci + 1) * chunk_size, total)
        if chunk_path.exists():
            z = np.load(chunk_path, allow_pickle=True)
            all_emb_parts.append(z["embeddings"].astype(np.float32))
            all_files.extend(z["files"].tolist())
            print(f"  Loaded cached chunk {ci}/{n_chunks-1}: {chunk_path.name}")
            continue
        subset_ds = torch.utils.data.Subset(dataset, list(range(start, end)))
        _pin      = device.type == "cuda"
        dl_kwargs: dict = dict(batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=_pin,
                               persistent_workers=(num_workers > 0))
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = 2
        loader    = DataLoader(subset_ds, **dl_kwargs)
        emb_np, files, _ = compute_embeddings(model, loader, device, amp)
        np.savez_compressed(chunk_path, embeddings=emb_np.astype(np.float32),
                            files=np.array(files, dtype=object))
        all_emb_parts.append(emb_np)
        all_files.extend(files)
        print(f"  Saved chunk {ci}/{n_chunks-1}: {chunk_path.name} ({len(files)} imgs)")
        del loader, emb_np, files, subset_ds
        free_gpu_memory()
    return np.vstack(all_emb_parts), all_files

# ══════════════════════════════════════════════════════════════════════════════
# DATA VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_data_integrity(
    data_dir: Path, train_df: pd.DataFrame, test_df: pd.DataFrame,
    sample_df: pd.DataFrame, train_root: Path, test_root: Path,
    strict_counts: bool = True,
) -> None:
    train_files   = {p.name for p in train_root.rglob("*.png")}
    test_files    = {p.name for p in test_root.rglob("*.png")}
    missing_train = sorted(set(train_df["filename"]) - train_files)
    missing_test_q = sorted(set(test_df["query_image"]) - test_files)
    missing_test_g = sorted(set(test_df["gallery_image"]) - test_files)
    assert not missing_train,   f"Missing train files: {missing_train[:5]}"
    assert not missing_test_q,  f"Missing test query files: {missing_test_q[:5]}"
    assert not missing_test_g,  f"Missing test gallery files: {missing_test_g[:5]}"
    assert test_df["row_id"].is_unique,   "row_id in test.csv must be unique"
    assert sample_df["row_id"].is_unique, "row_id in sample_submission.csv must be unique"
    assert set(test_df["row_id"]) == set(sample_df["row_id"]), "row_id mismatch: test vs sample"
    assert not train_df.duplicated(subset=["filename"]).any(), "Duplicate filenames in train.csv"
    if strict_counts:
        assert len(train_df) == 1895,             f"Expected 1895 train rows, got {len(train_df)}"
        assert train_df["ground_truth"].nunique() == 31, "Expected 31 train identities"
        assert len(test_files) == 371,            f"Expected 371 test images, got {len(test_files)}"
        assert len(test_df) == 137270,            f"Expected 137270 test rows, got {len(test_df)}"
    probe = train_root / train_df.iloc[0]["filename"]
    if not probe.exists():
        probe = list(train_root.rglob(train_df.iloc[0]["filename"]))[0]
    with Image.open(probe) as img:
        img.verify()

# ══════════════════════════════════════════════════════════════════════════════
# EDA
# ══════════════════════════════════════════════════════════════════════════════

def run_eda(data_dir: Path, output_dir: Path, train_df: pd.DataFrame,
            train_root: Path, seed: int, use_cache: bool = True) -> None:
    eda_dir = output_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    cdir = cache_dir(output_dir)
    manifest_path = cdir / "eda_manifest.json"
    eda_cfg = {"seed": int(seed), "data_dir": str(data_dir.resolve()),
               "train_csv": file_signature(data_dir / "train.csv"),
               "train_images": png_dir_signature(train_root)}
    eda_key = _json_hash(eda_cfg)
    if use_cache:
        prev     = load_manifest(manifest_path)
        required = [eda_dir / "eda_class_counts.csv", eda_dir / "eda_class_distribution.png",
                    eda_dir / "eda_image_dimensions.csv", eda_dir / "eda_size_aspect_hist.png",
                    eda_dir / "eda_random_identity_grid.png", eda_dir / "eda_dhash_table.csv",
                    eda_dir / "eda_near_duplicate_groups.csv", eda_dir / "eda_summary.json"]
        if prev and prev.get("eda_key") == eda_key and all(p.exists() for p in required):
            print(f"EDA cache hit ({eda_key}). Skipping.")
            return
    counts = train_df["ground_truth"].value_counts().sort_values(ascending=False)
    counts.to_csv(eda_dir / "eda_class_counts.csv", header=["count"])
    plt.figure(figsize=(12, 6)); counts.plot(kind="bar")
    plt.title("Train Class Distribution"); plt.xlabel("Jaguar Identity"); plt.ylabel("Images")
    plt.tight_layout(); plt.savefig(eda_dir / "eda_class_distribution.png", dpi=180); plt.close()
    dims = []
    for fn in tqdm(train_df["filename"].tolist(), desc="EDA image stats", leave=False):
        p = train_root / fn
        if not p.exists(): p = next(train_root.rglob(fn))
        with Image.open(p) as img: w, h = img.size
        dims.append((fn, w, h, w / max(h, 1)))
    dims_df = pd.DataFrame(dims, columns=["filename", "width", "height", "aspect"])
    dims_df.to_csv(eda_dir / "eda_image_dimensions.csv", index=False)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.hist(dims_df["width"], bins=30); plt.title("Width Histogram")
    plt.subplot(1, 2, 2); plt.hist(dims_df["aspect"], bins=30); plt.title("Aspect Ratio")
    plt.tight_layout(); plt.savefig(eda_dir / "eda_size_aspect_hist.png", dpi=180); plt.close()
    rnd = random.Random(seed)
    identities = train_df["ground_truth"].drop_duplicates().tolist()
    chosen_ids = rnd.sample(identities, k=min(9, len(identities)))
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    for i, gid in enumerate(chosen_ids):
        candidates = train_df[train_df["ground_truth"] == gid]["filename"].tolist()
        fn = rnd.choice(candidates)
        p  = train_root / fn
        if not p.exists(): p = next(train_root.rglob(fn))
        img = Image.open(p).convert("RGB")
        ax = axs[i // 3, i % 3]; ax.imshow(img); ax.set_title(gid); ax.axis("off")
    for j in range(len(chosen_ids), 9):
        axs[j // 3, j % 3].axis("off")
    plt.tight_layout(); plt.savefig(eda_dir / "eda_random_identity_grid.png", dpi=180); plt.close()
    hash_rows = []
    for fn in tqdm(train_df["filename"].tolist(), desc="EDA duplicate hash", leave=False):
        p = train_root / fn
        if not p.exists(): p = next(train_root.rglob(fn))
        with Image.open(p) as img: h = dhash(img)
        hash_rows.append((fn, h))
    hash_df = pd.DataFrame(hash_rows, columns=["filename", "dhash"])
    hash_df = hash_df.merge(train_df, on="filename", how="left")
    hash_df.to_csv(eda_dir / "eda_dhash_table.csv", index=False)
    dup_summary = (hash_df.groupby(["ground_truth", "dhash"]).size()
                   .reset_index(name="count").query("count > 1"))
    dup_summary.to_csv(eda_dir / "eda_near_duplicate_groups.csv", index=False)
    summary = {"data_dir": str(data_dir), "eda_cache_key": eda_key,
               "num_train_rows": int(len(train_df)),
               "num_classes": int(train_df["ground_truth"].nunique()),
               "min_images_per_class": int(counts.min()),
               "max_images_per_class": int(counts.max()),
               "estimated_near_duplicate_groups": int(len(dup_summary))}
    (eda_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    save_manifest(manifest_path, {"eda_key": eda_key, "config": eda_cfg,
                                  "outputs_dir": str(eda_dir.resolve())})

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING (fold-by-fold, resumable)
# ══════════════════════════════════════════════════════════════════════════════

def train_one_fold(
    args: argparse.Namespace,
    fold_id: int,
    train_df_full: pd.DataFrame,
    train_root: Path,
    label_to_idx: Dict[str, int],
    output_dir: Path,
    device: torch.device,
) -> Dict[str, object]:
    print(f"\n{'─'*60}\n FOLD {fold_id} | {gpu_mem_report()}\n{'─'*60}")
    folds  = build_stratified_image_folds(train_df_full, args.folds, args.seed)
    fold   = folds[fold_id]
    tr_df  = train_df_full.iloc[fold.train_idx].reset_index(drop=True)
    va_df  = train_df_full.iloc[fold.val_idx].reset_index(drop=True)

    model = JaguarReIDModel(
        args.backbone, num_classes=len(label_to_idx),
        embedding_dim=args.embedding_dim, img_size=args.img_size,
    ).to(device)
    effective_img_size = resolve_effective_img_size(model.backbone, args.img_size)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model, device)
    patch_timm_attention_for_mps(model, device)
    model = try_compile_model(model, device)

    effective_batch_size  = args.batch_size
    effective_num_workers = args.num_workers
    if device.type == "mps":
        effective_batch_size  = max(2, min(effective_batch_size, 4))
        effective_num_workers = min(effective_num_workers, 2)
    elif device.type == "cpu":
        effective_batch_size  = min(effective_batch_size, 8)
        effective_num_workers = min(effective_num_workers, 2)

    ds_tr = JaguarDataset(tr_df, train_root, make_train_transform(effective_img_size),
                          label_to_idx, use_alpha=args.use_alpha)
    ds_va = JaguarDataset(va_df, train_root, make_eval_transform(effective_img_size),
                          label_to_idx, use_alpha=args.use_alpha)
    labels_train = [label_to_idx[g] for g in tr_df["ground_truth"].tolist()]
    p = max(2, min(args.pk_identities, len(set(labels_train))))
    k = max(2, args.pk_instances)
    sampler = PKSampler(labels_train, p=p, k=k, length=len(ds_tr), seed=args.seed + fold_id)
    _pin = device.type == "cuda"
    dl_common: dict = dict(num_workers=effective_num_workers, pin_memory=_pin,
                           persistent_workers=(effective_num_workers > 0))
    if effective_num_workers > 0:
        dl_common["prefetch_factor"] = max(2, args.prefetch_factor)
    dl_tr = DataLoader(ds_tr, batch_size=effective_batch_size, sampler=sampler, drop_last=True, **dl_common)
    dl_va = DataLoader(ds_va, batch_size=effective_batch_size, shuffle=False, **dl_common)

    opt       = build_llrd_optimizer(model, lr=args.lr,
                    head_lr_multiplier=getattr(args, 'head_lr_multiplier', 10.0),
                    layer_decay=getattr(args, 'layer_decay', 0.8),
                    weight_decay=args.weight_decay)
    scaler    = create_grad_scaler(device, enabled=args.amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    class_weights = compute_class_weights(labels_train, num_classes=len(label_to_idx),
                                          mode=args.class_weighting, beta=args.effective_num_beta)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    if args.loss_type == "focal":
        ce_loss = FocalCrossEntropyLoss(class_weights=class_weights, gamma=args.focal_gamma,
                                        label_smoothing=args.label_smoothing)
    else:
        ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    grad_accum_steps = max(1, args.grad_accum_steps)
    if args.grad_accum_steps == 1 and effective_batch_size < args.batch_size:
        grad_accum_steps = max(1, args.batch_size // max(1, effective_batch_size))

    best_map, best_calib = -1.0, (1.0, 0.0)
    best_ckpt = output_dir / f"best_fold{fold_id}.pt"
    history: List[dict] = []

    es_monitor = EarlyStoppingMonitor(
        patience      = getattr(args, 'early_stopping_patience', 8),
        min_delta     = getattr(args, 'early_stopping_min_delta', 1e-4),
        warmup_epochs = getattr(args, 'early_stopping_warmup', 10),
        fold_id       = fold_id,
        output_dir    = output_dir,
    )
    print(f"  EarlyStopping: patience={es_monitor.patience} "
          f"min_delta={es_monitor.min_delta} "
          f"warmup={es_monitor.warmup_epochs} epochs")
    # ── Mid-fold resume: load epoch checkpoint if present ──────────────────
    start_epoch = 1
    epoch_ckpt  = output_dir / f"epoch_ckpt_fold{fold_id}.pt"
    if epoch_ckpt.exists():
        print(f"  Resuming fold {fold_id} from {epoch_ckpt.name} ...")
        resumed = torch.load(epoch_ckpt, map_location=device)
        model.load_state_dict(resumed["model_state_dict"])
        opt.load_state_dict(resumed["optimizer_state_dict"])
        scheduler.load_state_dict(resumed["scheduler_state_dict"])
        if resumed.get("scaler_state_dict") and hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(resumed["scaler_state_dict"])
        best_map    = resumed.get("best_map",   -1.0)
        best_calib  = resumed.get("best_calib", (1.0, 0.0))
        history     = resumed.get("history",    [])
        start_epoch = resumed["epoch"] + 1
        # Restore EarlyStopping state so patience counter isn't reset
        es_monitor.best_map   = best_map
        es_monitor.best_epoch = resumed["epoch"]
        completed_epochs = [h for h in history if not math.isnan(h.get("val_id_balanced_map", float("nan")))]
        if completed_epochs:
            last_improving = max((h["epoch"] for h in completed_epochs
                                  if h["val_id_balanced_map"] >= best_map - es_monitor.min_delta),
                                 default=resumed["epoch"])
            es_monitor.wait = resumed["epoch"] - last_improving
        print(f"  Resumed: start_epoch={start_epoch}, best_map={best_map:.5f}, "
              f"es_wait={es_monitor.wait}/{es_monitor.patience}")
        del resumed
    # ───────────────────────────────────────────────────────────────────────

    for epoch in range(start_epoch, args.epochs + 1):   # ← start_epoch not 1
        model.train()
        tr_loss, n_steps = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for x, y, _ in tqdm(dl_tr, desc=f"F{fold_id} E{epoch} train", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast_context(device=device, enabled=args.amp):
                emb, logits = model(x, y)
                loss_ce  = ce_loss(logits, y)
                loss_tri = batch_hard_triplet_loss(emb, y) if args.triplet_weight > 0 else 0.0
                loss     = (loss_ce + args.triplet_weight * loss_tri) / grad_accum_steps
            scaler.scale(loss).backward()
            if (n_steps + 1) % grad_accum_steps == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            tr_loss += float(loss.item()); n_steps += 1
        if n_steps % grad_accum_steps != 0:
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        scheduler.step()
        tr_loss = (tr_loss * grad_accum_steps) / max(1, n_steps)

        should_validate = (epoch % max(1, args.val_every_epochs) == 0) or (epoch == args.epochs)
        if not should_validate:
            history.append({"fold": fold_id, "epoch": epoch, "train_loss": tr_loss, "val_id_balanced_map": np.nan})
            print(f"  [F{fold_id}][E{epoch}] loss={tr_loss:.5f} mAP=skip {gpu_mem_report()}")
            continue

        emb_va, files_va, _ = compute_embeddings(model, dl_va, device, args.amp)
        y_names = np.array([va_df.iloc[i]["ground_truth"] for i in range(len(files_va))])
        val_map, _ = identity_balanced_map(emb_va, y_names)
        history.append({"fold": fold_id, "epoch": epoch, "train_loss": tr_loss, "val_id_balanced_map": val_map})
        print(f"  [F{fold_id}][E{epoch}] loss={tr_loss:.5f} mAP={val_map:.5f} {gpu_mem_report()}")

        if val_map > best_map + es_monitor.min_delta:  # already-instantiated monitor
            best_map = val_map
            best_calib = (1.0, 0.0)

            # ── Save 1: clean inference checkpoint ───────────────────────────────
            ckpt_payload = {
                "model_state_dict": model.state_dict(),
                "backbone": args.backbone,
                "embedding_dim": args.embedding_dim,
                "feat_dim": model.bn.num_features,
                "img_size": effective_img_size,
                "label_to_idx": label_to_idx,
                "best_map": best_map,
                "calibration_a": best_calib[0],
                "calibration_b": best_calib[1],
            }
            torch.save(ckpt_payload, best_ckpt)
            print(f"\n** Save1: ** New best mAP={best_map:.5f} → saved {best_ckpt.name} **\n")
        # ── Save 2: full training state (every epoch, not just best) ─────────────
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if hasattr(scaler, "load_state_dict") else {},
            "best_map": best_map,
            "best_calib": best_calib,
            "history": history,
        }, epoch_ckpt)
        print(f"\n** Save 2: Full training state (every epoch, not just best) cached at {epoch_ckpt} **\n")


        # ── Early stopping check ──────────────────────────────────────────────
        if es_monitor.step(epoch, val_map):
            print(f"  Restoring best weights from {best_ckpt}")
            ckpt_best = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(ckpt_best["model_state_dict"])
            del ckpt_best
            break  # exit epoch loop — best weights now loaded


    pd.DataFrame(history).to_csv(output_dir / f"train_history_fold{fold_id}.csv", index=False)
    del model, opt, scaler, scheduler, ce_loss, dl_tr, dl_va, ds_tr, ds_va
    free_gpu_memory()
    print(f"  Fold {fold_id} done. best_mAP={best_map:.5f} {gpu_mem_report()}")
    return {"fold_id": fold_id, "best_map": best_map,
            "checkpoint": str(best_ckpt), "calibration": best_calib}

def train_pipeline_phased(
    args: argparse.Namespace, data_dir: Path, output_dir: Path,
    train_df: pd.DataFrame, train_root: Path, device: torch.device,
) -> Tuple[Path, List[Path]]:
    """Returns (best_checkpoint, all_fold_checkpoints)."""
    labels       = sorted(train_df["ground_truth"].unique().tolist())
    label_to_idx = {l: i for i, l in enumerate(labels)}
    fold_results: List[dict] = []
    for fold_id in range(args.folds):
        phase_name = f"train_fold_{fold_id}"
        ckpt_path  = output_dir / f"best_fold{fold_id}.pt"
        if is_phase_complete(output_dir, phase_name) and ckpt_path.exists():
            print(f"\n  Phase '{phase_name}' already complete — skipping.")
            prev = load_manifest(phase_marker_path(output_dir, phase_name))
            fold_results.append({"fold_id": fold_id,
                                  "best_map": prev.get("best_map", 0.0) if prev else 0.0,
                                  "checkpoint": str(ckpt_path),
                                  "calibration": (1.0, 0.0)})
            continue
        result = train_one_fold(args, fold_id, train_df, train_root, label_to_idx, output_dir, device)
        fold_results.append(result)
        mark_phase_complete(output_dir, phase_name, {"best_map": result["best_map"]})
    fold_df   = pd.DataFrame(fold_results).sort_values("best_map", ascending=False)
    fold_df.to_csv(output_dir / "cv_fold_results.csv", index=False)
    best_ckpt = Path(fold_df.iloc[0]["checkpoint"])
    all_ckpts = [Path(r["checkpoint"]) for r in fold_results]
    print(f"\n  Best checkpoint across folds: {best_ckpt}")
    return best_ckpt, all_ckpts


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE  (multi-fold ensemble · TTA · k-reciprocal re-ranking)
# All embedding steps are individually cached and fully resumable.
# ══════════════════════════════════════════════════════════════════════════════

def _load_model_for_inference(
    checkpoint_path: Path, args: argparse.Namespace, device: torch.device,
) -> Tuple["nn.Module", int, float, float]:
    """Load one fold checkpoint → (model, effective_img_size, calib_a, calib_b)."""
    ckpt         = torch.load(checkpoint_path, map_location="cpu")
    label_to_idx = ckpt["label_to_idx"]
    model = JaguarReIDModel(
        ckpt.get("backbone", args.backbone),
        num_classes=len(label_to_idx),
        embedding_dim=ckpt.get("embedding_dim", args.embedding_dim),
        img_size=ckpt.get("img_size", args.img_size),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    effective_img_size = resolve_effective_img_size(model.backbone, args.img_size)
    if device.type == "mps" and effective_img_size > 336:
        print(f"  MPS: capping img_size {effective_img_size} → 336.")
        effective_img_size = 336
    patch_timm_attention_for_mps(model, device)
    model = try_compile_model(model, device)
    a = float(ckpt.get("calibration_a", 1.0))
    b = float(ckpt.get("calibration_b", 0.0))
    del ckpt; free_gpu_memory()
    return model, effective_img_size, a, b


def extract_fold_tta_embeddings(
    fold_id: int,
    checkpoint_path: Path,
    args: argparse.Namespace,
    test_root: Path,
    unique_test: List[str],
    device: torch.device,
    cdir: Path,
) -> np.ndarray:
    """
    Extract TTA-averaged embeddings for ONE fold checkpoint.
    Per-TTA-augmentation chunk files are cached individually.
    The final TTA average for this fold is cached as a single .npz.
    On resume, the fold-level .npz is loaded directly — no GPU work needed.
    """
    ckpt_stem    = checkpoint_path.stem                          # e.g. "best_fold2"
    ensemble_npz = cdir / f"test_emb_{ckpt_stem}_tta_ensemble.npz"

    if ensemble_npz.exists():
        print(f"  Fold {fold_id}: loading cached TTA ensemble ← {ensemble_npz.name}")
        z = np.load(ensemble_npz, allow_pickle=True)
        return z["embeddings"].astype(np.float32)

    print(f"\n  Fold {fold_id}: extracting embeddings from {checkpoint_path.name} ...")
    model, effective_img_size, _, _ = _load_model_for_inference(checkpoint_path, args, device)
    print(f"  Model loaded. {gpu_mem_report()}")

    test_img_df    = pd.DataFrame({"filename": unique_test})
    _base_tf       = make_eval_transform(effective_img_size)
    _hflip_tf      = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), _base_tf])
    _vflip_tf      = transforms.Compose([transforms.RandomVerticalFlip(p=1.0),   _base_tf])
    _both_tf       = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0),
                                          transforms.RandomVerticalFlip(p=1.0),   _base_tf])
    tta_transforms = [_base_tf, _hflip_tf, _vflip_tf, _both_tf]
    test_workers   = min(args.num_workers, 2) if device.type == "mps" else args.num_workers

    tta_embs: List[np.ndarray] = []
    files_test: List[str]      = []
    for ti, tf in enumerate(tta_transforms):
        ds_tta       = JaguarDataset(test_img_df, test_root, tf,
                                     label_to_idx=None, use_alpha=args.use_alpha)
        cache_prefix = cdir / f"test_emb_{ckpt_stem}_tta{ti}"   # per-fold, per-aug cache
        e, files_test = compute_embeddings_chunked(
            model=model, dataset=ds_tta, device=device, amp=args.amp,
            batch_size=(max(2, min(args.batch_size, 4)) if device.type == "mps"
                        else args.batch_size),
            num_workers=test_workers, chunk_size=args.embed_chunk_size,
            cache_prefix=cache_prefix,
        )
        tta_embs.append(e)
        del ds_tta

    emb_fold = np.mean(np.stack(tta_embs, axis=0), axis=0)
    print(f"  Fold {fold_id}: TTA averaged ({len(tta_transforms)} augmentations).")
    np.savez_compressed(ensemble_npz,
                        embeddings=emb_fold.astype(np.float32),
                        files=np.array(files_test, dtype=object))
    print(f"  Fold {fold_id}: TTA ensemble cached → {ensemble_npz.name}")

    del model; free_gpu_memory()
    print(f"  {gpu_mem_report()}")
    return emb_fold


def extract_test_ensemble(
    args: argparse.Namespace,
    fold_checkpoints: List[Path],
    test_root: Path,
    output_dir: Path,
    device: torch.device,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract (or reload) per-fold TTA embeddings, then average across all folds.
    The final cross-fold ensemble is cached as test_emb_ensemble_final.npz.
    On full resume this function returns in milliseconds — zero GPU work.
    """
    cdir      = cache_dir(output_dir)
    final_npz = cdir / "test_emb_ensemble_final.npz"

    if final_npz.exists():
        print(f"\n  Loading cached final ensemble ← {final_npz.name}")
        z = np.load(final_npz, allow_pickle=True)
        return z["embeddings"].astype(np.float32), list(z["files"])

    print(f"\n══ Phase: Embedding ({len(fold_checkpoints)} folds × 4 TTA) ══")
    test_df_    = pd.read_csv(args.data_dir / "test.csv")
    unique_test = sorted(set(test_df_["query_image"]).union(set(test_df_["gallery_image"])))

    fold_embs: List[np.ndarray] = []
    files_test: List[str]       = []
    for fold_id, ckpt_path in enumerate(fold_checkpoints):
        emb = extract_fold_tta_embeddings(
            fold_id, ckpt_path, args, test_root, unique_test, device, cdir
        )
        fold_embs.append(emb)
        # recover files list from this fold's cached npz
        last_npz   = cdir / f"test_emb_{ckpt_path.stem}_tta_ensemble.npz"
        files_test = list(np.load(last_npz, allow_pickle=True)["files"])
        free_gpu_memory()

    ensemble_emb = np.mean(np.stack(fold_embs, axis=0), axis=0)
    print(f"\n  Ensemble: averaged {len(fold_embs)} fold embeddings → shape {ensemble_emb.shape}")
    np.savez_compressed(final_npz,
                        embeddings=ensemble_emb.astype(np.float32),
                        files=np.array(files_test, dtype=object))
    print(f"  Final ensemble cached → {final_npz.name}")
    return ensemble_emb, files_test


def inference_pipeline_chunked(
    args: argparse.Namespace, data_dir: Path, output_dir: Path,
    test_df: pd.DataFrame, test_root: Path,
    fold_checkpoints: List[Path],
    device: torch.device,
) -> Path:
    """
    Full inference pipeline — every embedding step is cached and resumable.
    Phase map:
      embed  → test_emb_ensemble_final.npz + phase_embed_done.json
      submit → submission.csv              + phase_submit_done.json
    """
    out_path = output_dir / "submission.csv"

    # Calibration from the best (first) checkpoint
    ckpt_meta = torch.load(fold_checkpoints[0], map_location="cpu")
    a = float(ckpt_meta.get("calibration_a", 1.0))
    b = float(ckpt_meta.get("calibration_b", 0.0))
    del ckpt_meta; gc.collect()

    # ── Step 1: ensemble embeddings (fully cached/resumable) ─────────────────
    emb_test, files_test = extract_test_ensemble(
        args, fold_checkpoints, test_root, output_dir, device
    )
    mark_phase_complete(output_dir, "embed", {"folds": len(fold_checkpoints)})

    # ── Step 2: L2 normalise ─────────────────────────────────────────────────
    norms    = np.linalg.norm(emb_test, axis=1, keepdims=True)
    emb_test = emb_test / np.clip(norms, 1e-8, None)

    # After L2 normalise, before re-ranking
    if args.use_query_expansion:
        emb_test = query_expansion(emb_test, top_k=args.qe_top_k)

    # ── Step 3: k-reciprocal re-ranking ──────────────────────────────────────
    print("  Applying k-reciprocal re-ranking (k1=20, k2=6, λ=0.3)…")
    sim_mat = k_reciprocal_rerank(emb_test, k1=20, k2=6, lam=0.3)
    sim_mat = 0.5 * (sim_mat + sim_mat.T)

    file_to_idx = {f: i for i, f in enumerate(files_test)}
    q_idx = test_df["query_image"].map(file_to_idx).fillna(-1).astype(np.int32).to_numpy()
    g_idx = test_df["gallery_image"].map(file_to_idx).fillna(-1).astype(np.int32).to_numpy()
    if np.any(q_idx < 0) or np.any(g_idx < 0):
        raise ValueError("Failed to map one or more query/gallery filenames to embedding indices.")
    sims = sim_mat[q_idx, g_idx].astype(np.float32)
    del sim_mat, emb_test; gc.collect()

    # ── Step 4: calibrate + write submission ─────────────────────────────────
    probs = apply_calibration(sims, a, b)
    sub   = pd.DataFrame({"row_id": test_df["row_id"].astype(int),
                           "similarity": probs.astype(float)})
    sub   = sub.sort_values("row_id").reset_index(drop=True)
    assert len(sub) == len(test_df),                                   "Submission row count mismatch"
    assert sub["row_id"].tolist() == sorted(test_df["row_id"].tolist()), "row_id order mismatch"
    assert np.isfinite(sub["similarity"].values).all(),                 "Non-finite similarity found"
    assert ((sub["similarity"].values >= 0.0) & (sub["similarity"].values <= 1.0)).all()
    sub.to_csv(out_path, index=False)
    print(f"  Wrote submission to: {out_path}")
    return out_path



# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION CONSTANTS  — edit these before running
# ══════════════════════════════════════════════════════════════════════════════
# ── PATHS ── update these two lines ──────────────────────────
DATA_DIR   = Path(r"D:\datascience\kaggle_large_datasets\jaguar_id\data")
OUTPUT_DIR = Path(r"D:\datascience\kaggle_large_datasets\jaguar_id\output")


# ── Kaggle-specific stuff ── leave OFF locally ────────────────
INSTALL_DEPS = False
GCS_DATA_URI = None
GCS_OUTPUT_URI = None

# ── Phase control ── KEEP as-is ──────────────────────────────
PHASE = "auto"   # pipeline will auto-skip completed phases
USE_CACHE = True
STRICT_COUNTS   = True
CHECKPOINT_PATH = None
"""
# ── Paths (update for your environment) ──────────────────────────────────────
DATA_DIR   = Path("/kaggle/input/competitions/jaguar-re-id")  # adjust to your dataset\n
OUTPUT_DIR = Path("/kaggle/working")

# ── GCP / Kaggle ──────────────────────────────────────────────────────────────
INSTALL_DEPS   = False
GCS_DATA_URI   = None
GCS_OUTPUT_URI = None

# ── Pipeline Control ──────────────────────────────────────────────────────────
# ⚠  If previous run used broken MPS backward, DELETE output/cache/ first!
PHASE           = "auto"
CHECKPOINT_PATH = None
USE_CACHE       = True
STRICT_COUNTS   = True
"""
# ── Defaults (ImageNet — safe fallback for most timm backbones) ──────────────
NORM_MEAN     = [0.485, 0.456, 0.406]
NORM_STD      = [0.229, 0.224, 0.225]

# ── Backbone Selection (DINOv2 vs MegaDescriptor) ────────────────────────────
BACKBONE_CHOICE = "mega"  # "dino" or "mega" — competition suggested MegaDescriptor

# ── ML Hyperparameters (auto-adjusted for backbone) ──────────────────────────
if BACKBONE_CHOICE == "dino":
    BACKBONE      = "vit_base_patch14_dinov2.lvd142m"
    IMG_SIZE      = 518
    EMBEDDING_DIM = 768  # matches ViT-B/14
elif BACKBONE_CHOICE == "mega":
    BACKBONE      = "hf-hub:BVRA/MegaDescriptor-L-384"
    IMG_SIZE      = 384
    EMBEDDING_DIM = 1024  # Swin-L typical output
    NORM_MEAN     = [0.5, 0.5, 0.5]
    NORM_STD      = [0.5, 0.5, 0.5]
else:
    raise ValueError(f"Unknown BACKBONE_CHOICE: {BACKBONE_CHOICE}")

# ── Rest of hyperparameters (high-compute, no constraints) ───────────────────
BATCH_SIZE        = 16            # doubled vs v3; effective batch = 16*2 = 32
EPOCHS            = 40            # ~1600 grad steps/fold; ArcFace converges well
FOLDS             = 5             # 80% train / 20% val per fold
LR                = 5e-5          # lower LR for stable ViT fine-tuning
WEIGHT_DECAY      = 1e-4
LABEL_SMOOTHING   = 0.10          # more regularisation for 31-class problem
TRIPLET_WEIGHT    = 0.30          # stronger metric learning signal
LOSS_TYPE         = "focal"
FOCAL_GAMMA       = 2.0
CLASS_WEIGHTING   = "effective_num"
EFFECTIVE_NUM_BETA = 0.999
PK_IDENTITIES   = 4    # 4 identities × 4 instances = 16
PK_INSTANCES    = 4

SEED              = 42
#NUM_WORKERS       = 2           # Kaggle allows up to 2 reliably
NUM_WORKERS       = 0            # ✅ Safe on Windows — avoids worker spawn overhead causing slowdowns
PREFETCH_FACTOR   = 2
GRAD_ACCUM_STEPS  = 2            # effective batch = 16 * 2 = 32
VAL_EVERY_EPOCHS  = 1            # validate every epoch to catch peak early
USE_ALPHA         = True         # alpha mask focuses model on jaguar body
AMP               = True

# ── Memory knobs ──────────────────────────────────────────────────────────────
GRADIENT_CHECKPOINTING = True     # halves backbone activation memory on CUDA
EMBED_CHUNK_SIZE       = 100      # images per embedding chunk

def build_config() -> argparse.Namespace:
    return argparse.Namespace(
        data_dir           = DATA_DIR,
        output_dir         = OUTPUT_DIR,
        install_deps       = INSTALL_DEPS,
        gcs_data_uri       = GCS_DATA_URI,
        gcs_output_uri     = GCS_OUTPUT_URI,
        phase              = PHASE,
        checkpoint_path    = Path(CHECKPOINT_PATH) if CHECKPOINT_PATH else None,
        use_cache          = USE_CACHE,
        strict_counts      = STRICT_COUNTS,
        backbone           = BACKBONE,
        img_size           = IMG_SIZE,
        norm_mean          = NORM_MEAN,    # NEW: dynamic for DINOv2/MegaDescriptor
        norm_std           = NORM_STD,     # NEW: dynamic for DINOv2/MegaDescriptor
        embedding_dim      = EMBEDDING_DIM,
        batch_size         = BATCH_SIZE,
        epochs             = EPOCHS,
        folds              = FOLDS,
        lr                 = LR,
        weight_decay       = WEIGHT_DECAY,
        label_smoothing    = LABEL_SMOOTHING,
        triplet_weight     = TRIPLET_WEIGHT,
        loss_type          = LOSS_TYPE,
        focal_gamma        = FOCAL_GAMMA,
        class_weighting    = CLASS_WEIGHTING,
        effective_num_beta = EFFECTIVE_NUM_BETA,
        pk_identities      = PK_IDENTITIES,
        pk_instances       = PK_INSTANCES,
        seed               = SEED,
        num_workers        = NUM_WORKERS,
        prefetch_factor    = PREFETCH_FACTOR,
        grad_accum_steps   = GRAD_ACCUM_STEPS,
        val_every_epochs   = VAL_EVERY_EPOCHS,
        use_query_expansion = True,
        early_stopping_patience = 8,       # eval epochs without improvement
        early_stopping_min_delta = 1e-4,   # minimum meaningful improvement
        early_stopping_warmup   = 10,      # don't stop before this epoch

        qe_top_k           = 5,
        use_rerank         = True,
        head_lr_multiplier = 10.0,
        layer_decay        = 0.8,
        use_alpha          = USE_ALPHA,
        amp                = AMP,
        gradient_checkpointing = GRADIENT_CHECKPOINTING,
        embed_chunk_size   = EMBED_CHUNK_SIZE,
    )
def main() -> None:
    print_gpu_info()
    hf_connect()
    args = build_config()
    if args.install_deps:
        install_dependencies()
    data_dir   = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir   = data_dir
    args.output_dir = output_dir
    if args.gcs_data_uri:
        print(f"Syncing from GCS: {args.gcs_data_uri}")
        gcs_pull(args.gcs_data_uri, data_dir)
    set_seed(args.seed)

    import warnings
    warnings.filterwarnings("ignore", message=".*max_autotune_gemm.*")
    import logging
    logging.getLogger("torch._inductor").setLevel(logging.ERROR)

    train_csv  = data_dir / "train.csv"
    test_csv   = data_dir / "test.csv"
    sample_csv = data_dir / "sample_submission.csv"
    if not all(f.exists() for f in [train_csv, test_csv, sample_csv]):
        raise FileNotFoundError(f"Missing expected CSVs in {data_dir}")
    train_root = resolve_image_root(data_dir, "train")
    test_root  = resolve_image_root(data_dir, "test")
    print(f"Train root: {train_root}")
    print(f"Test root:  {test_root}")
    train_df   = pd.read_csv(train_csv)
    test_df    = pd.read_csv(test_csv)
    sample_df  = pd.read_csv(sample_csv)
    validate_data_integrity(data_dir, train_df, test_df, sample_df,
                            train_root, test_root, strict_counts=args.strict_counts)
    device = select_best_device()
    print(f"Device: {device} {gpu_mem_report()}")
    if device.type != "cuda" and args.amp:
        print("AMP requires CUDA. Disabling.")
        args.amp = False
    args.num_workers = resolve_num_workers(args.num_workers, device)
    if device.type == "mps" and args.img_size > 336:
        print(f"  MPS: img_size {args.img_size} → 336 (set before model build).")
        args.img_size = 336

    phase = args.phase.lower().strip()

    def should_run(name: str) -> bool:
        if phase == "auto": return True
        if phase == name:   return True
        if phase == "train" and name.startswith("train_fold_"): return True
        return False

    if should_run("eda"):
        if not is_phase_complete(output_dir, "eda"):
            print("\n══ Phase: EDA ══")
            run_eda(data_dir, output_dir, train_df, train_root, args.seed, use_cache=args.use_cache)
            mark_phase_complete(output_dir, "eda")
            free_gpu_memory()
        else:
            print("\n  Phase 'eda' already complete — skipping.")

    checkpoint = args.checkpoint_path
    all_fold_checkpoints: List[Path] = []

    if should_run("train") or any(should_run(f"train_fold_{i}") for i in range(args.folds)):
        print("\n══ Phase: Training ══")
        checkpoint, all_fold_checkpoints = train_pipeline_phased(
            args, data_dir, output_dir, train_df, train_root, device
        )
        mark_phase_complete(output_dir, "train", {"checkpoint": str(checkpoint)})
        free_gpu_memory()

    if should_run("embed") or should_run("submit"):
        # Resolve fold checkpoints if we skipped training
        if not all_fold_checkpoints:
            cv_results = output_dir / "cv_fold_results.csv"
            if cv_results.exists():
                fold_df = pd.read_csv(cv_results)
                all_fold_checkpoints = [Path(p) for p in fold_df["checkpoint"].tolist()]
            elif checkpoint:
                all_fold_checkpoints = [checkpoint]
            else:
                raise ValueError("No checkpoints available. Run training first.")

        if not is_phase_complete(output_dir, "submit"):
            print("\n══ Phase: Embed + Submit ══")
            inference_pipeline_chunked(
                args, data_dir, output_dir, test_df, test_root,
                all_fold_checkpoints, device
            )
            mark_phase_complete(output_dir, "submit")
            free_gpu_memory()
        else:
            print("\n  Phase 'submit' already complete — skipping.")

    if args.gcs_output_uri:
        print(f"Uploading to GCS: {output_dir} -> {args.gcs_output_uri}")
        gcs_push(output_dir, args.gcs_output_uri)

    print(f"\nPipeline complete. Output: {output_dir}")
    sub = output_dir / "submission.csv"
    if sub.exists():
        print(f"Submission CSV: {sub}")

#Lets go!!!
# ✅ Fixed — required on Windows for DataLoader multiprocessing
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()