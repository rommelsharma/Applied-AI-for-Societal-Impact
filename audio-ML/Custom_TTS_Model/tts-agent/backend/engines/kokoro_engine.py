"""Kokoro-82M TTS engine wrapper (Apache 2.0 licensed — commercial-safe)."""
import io
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.engines.base import TTSEngine
from backend.utils.audio import postprocess, audio_to_bytes
from backend.utils.logger import get_logger
from configs.settings import settings

logger = get_logger(__name__)

_KOKORO_REPO_ID = "hexgrad/Kokoro-82M"


def _configure_espeak() -> None:
    """
    Point espeak-ng at a usable data directory before KPipeline / phonemizer
    are first imported.

    Why this is needed
    ------------------
    espeakng_loader calls espeak_ng_InitializePath() at import time.  That C
    function has an internal snprintf(buf, 160, …) buffer; a venv path longer
    than ~140 chars is silently truncated, both directory checks fail, and the
    library falls through to its compiled-in CI path
    ('/Users/runner/work/...') which does not exist on the user's machine.

    Strategy (tried in order)
    -------------------------
    1. System espeak-ng installed via `brew install espeak-ng`
       → data at /opt/homebrew/share/espeak-ng-data (short path, reliable).
    2. Short /tmp symlink → espeakng_loader/espeak-ng-data
       → workaround for the 160-char buffer limit when venv path is long.
    3. espeakng_loader path as-is (only works for short venv paths).

    On macOS the strongly recommended approach is:
        brew install espeak-ng
    """
    # ------------------------------------------------------------------
    # 1. Prefer system espeak-ng (Homebrew on macOS / system on Linux)
    #    Homebrew installs data to PREFIX/share/espeak-ng-data, not lib.
    # ------------------------------------------------------------------
    system_candidates = [
        "/opt/homebrew/share/espeak-ng-data",  # Apple Silicon Homebrew
        "/usr/local/share/espeak-ng-data",      # Intel Mac Homebrew
        "/usr/share/espeak-ng-data",            # Debian/Ubuntu/Fedora
        "/usr/lib/espeak-ng-data",              # some older Linux distros
        "/opt/homebrew/lib/espeak-ng-data",     # symlink Homebrew creates
        "/usr/local/lib/espeak-ng-data",        # symlink on Intel Mac
    ]
    for candidate in system_candidates:
        p = Path(candidate)
        if p.is_dir() and (p / "phontab").exists():
            data_path = candidate
            logger.debug("Using system espeak-ng data: %s", data_path)
            _apply_espeak_path(data_path)
            return

    # ------------------------------------------------------------------
    # 2. Fall back to espeakng_loader (bundled), with short-path workaround
    #
    # espeakng_loader.__init__.py does NOT call espeak_ng_InitializePath()
    # at import time — it just loads the dylib and provides helpers.
    # The C initialisation happens inside phonemizer's EspeakWrapper when
    # it first uses the library, so setting ESPEAK_DATA_PATH and calling
    # EspeakWrapper.set_data_path() here (before any kokoro/phonemizer
    # import) is effective.
    #
    # Bug: get_data_path() returns  .../espeakng_loader/espeak-ng-data
    # which is ~189 chars on typical macOS paths — well over the 160-char
    # snprintf buffer in espeak_ng_InitializePath().  Fix: symlink to /tmp.
    # ------------------------------------------------------------------
    try:
        import importlib.util as _ilu
        spec = _ilu.find_spec("espeakng_loader")
        if spec is None:
            raise ImportError("espeakng_loader not found")
        # Use the espeak-ng-data subdirectory, not the package root
        pkg_dir = Path(spec.origin).parent
        data_path = str(pkg_dir / "espeak-ng-data")
    except Exception as exc:
        logger.warning("espeakng_loader not available: %s", exc)
        return

    if len(data_path) > 140:
        short = Path("/tmp/kokoro_espeak_data")
        try:
            real = Path(data_path).resolve()
            if short.is_symlink() and short.resolve() != real:
                short.unlink()
            if not short.exists():
                short.symlink_to(str(real))
            data_path = str(short)
            logger.debug("espeak data path aliased to short symlink: %s", data_path)
        except Exception as exc:
            logger.warning("Could not create espeak data symlink: %s — trying raw path", exc)

    _apply_espeak_path(data_path)


def _apply_espeak_path(data_path: str) -> None:
    """Set the espeak data path everywhere it needs to be set.

    Critical detail: misaki/espeak.py (imported by kokoro at module level)
    runs these two lines at import time:
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

    The second call overwrites anything we set earlier.  To prevent this we
    monkey-patch espeakng_loader.get_data_path so that when misaki imports it,
    it gets our short correct path instead of the long venv path.
    """
    os.environ["ESPEAK_DATA_PATH"] = data_path
    os.environ["PHONEMIZER_ESPEAK_DATA_PATH"] = data_path

    # Monkey-patch espeakng_loader before kokoro/misaki can import it
    try:
        import espeakng_loader as _el
        _final_path = data_path  # capture for lambda
        _el.get_data_path = lambda: _final_path
        logger.debug("espeakng_loader.get_data_path patched → %s", data_path)
    except Exception as exc:
        logger.warning("Could not patch espeakng_loader: %s", exc)

    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_data_path(data_path)
        logger.debug("EspeakWrapper.set_data_path → %s", data_path)
    except Exception as exc:
        logger.warning("Could not set EspeakWrapper data path: %s", exc)


# Configure espeak at import time — must happen before KPipeline is instantiated
_configure_espeak()

# Voice catalog: id → (display name, lang_code for KPipeline)
#
# Grades from hexgrad/Kokoro-82M VOICES.md (quality × training data):
#   A/A- = production-ready   B/B- = good   C+/C = usable   D/F = experimental
#
# Language constraints
# --------------------
# Each voice is tied to its language pipeline — an English voice cannot read
# Hindi text and vice versa.  Use the voice whose language matches your script.
# Japanese (j) and Mandarin (z) require:  pip install misaki[ja]  /  misaki[zh]
#
_VOICES: dict[str, dict] = {
    # ── American English (lang_code='a') ──────────────────────────────────
    "af_heart":   {"name": "Heart ★ (EN-US Female)",   "language": "en-us", "lang_code": "a"},  # A
    "af_bella":   {"name": "Bella ★ (EN-US Female)",   "language": "en-us", "lang_code": "a"},  # A-
    "af_aoede":   {"name": "Aoede (EN-US Female)",     "language": "en-us", "lang_code": "a"},  # C+
    "af_kore":    {"name": "Kore (EN-US Female)",      "language": "en-us", "lang_code": "a"},  # C+
    "af_alloy":   {"name": "Alloy (EN-US Female)",     "language": "en-us", "lang_code": "a"},  # C
    "af_nova":    {"name": "Nova (EN-US Female)",      "language": "en-us", "lang_code": "a"},  # C
    "af_nicole":  {"name": "Nicole (EN-US Female)",    "language": "en-us", "lang_code": "a"},  # B-
    "af_sarah":   {"name": "Sarah (EN-US Female)",     "language": "en-us", "lang_code": "a"},  # C+
    "af_sky":     {"name": "Sky (EN-US Female)",       "language": "en-us", "lang_code": "a"},  # C-
    "af_jessica": {"name": "Jessica (EN-US Female)",   "language": "en-us", "lang_code": "a"},  # D
    "af_river":   {"name": "River (EN-US Female)",     "language": "en-us", "lang_code": "a"},  # D
    "am_fenrir":  {"name": "Fenrir ★ (EN-US Male)",    "language": "en-us", "lang_code": "a"},  # C+
    "am_michael": {"name": "Michael ★ (EN-US Male)",   "language": "en-us", "lang_code": "a"},  # C+
    "am_puck":    {"name": "Puck ★ (EN-US Male)",      "language": "en-us", "lang_code": "a"},  # C+
    "am_echo":    {"name": "Echo (EN-US Male)",        "language": "en-us", "lang_code": "a"},  # D
    "am_eric":    {"name": "Eric (EN-US Male)",        "language": "en-us", "lang_code": "a"},  # D
    "am_liam":    {"name": "Liam (EN-US Male)",        "language": "en-us", "lang_code": "a"},  # D
    "am_onyx":    {"name": "Onyx (EN-US Male)",        "language": "en-us", "lang_code": "a"},  # D
    "am_adam":    {"name": "Adam (EN-US Male)",        "language": "en-us", "lang_code": "a"},  # F+
    "am_santa":   {"name": "Santa (EN-US Male)",       "language": "en-us", "lang_code": "a"},  # D-
    # ── British English (lang_code='b') ───────────────────────────────────
    "bf_emma":    {"name": "Emma ★ (EN-GB Female)",    "language": "en-gb", "lang_code": "b"},  # B-
    "bf_isabella":{"name": "Isabella (EN-GB Female)",  "language": "en-gb", "lang_code": "b"},  # C
    "bf_alice":   {"name": "Alice (EN-GB Female)",     "language": "en-gb", "lang_code": "b"},  # D
    "bf_lily":    {"name": "Lily (EN-GB Female)",      "language": "en-gb", "lang_code": "b"},  # D
    "bm_george":  {"name": "George ★ (EN-GB Male)",    "language": "en-gb", "lang_code": "b"},  # C
    "bm_fable":   {"name": "Fable (EN-GB Male)",       "language": "en-gb", "lang_code": "b"},  # C
    "bm_daniel":  {"name": "Daniel (EN-GB Male)",      "language": "en-gb", "lang_code": "b"},  # D
    "bm_lewis":   {"name": "Lewis (EN-GB Male)",       "language": "en-gb", "lang_code": "b"},  # D+
    # ── Hindi (lang_code='h') ─────────────────────────────────────────────
    "hf_alpha":   {"name": "Alpha (HI Female)",        "language": "hi",    "lang_code": "h"},  # C
    "hf_beta":    {"name": "Beta (HI Female)",         "language": "hi",    "lang_code": "h"},  # C
    "hm_omega":   {"name": "Omega (HI Male)",          "language": "hi",    "lang_code": "h"},  # C
    "hm_psi":     {"name": "Psi (HI Male)",            "language": "hi",    "lang_code": "h"},  # C
    # ── Spanish (lang_code='e') ───────────────────────────────────────────
    "ef_dora":    {"name": "Dora (ES Female)",         "language": "es",    "lang_code": "e"},
    "em_alex":    {"name": "Alex (ES Male)",           "language": "es",    "lang_code": "e"},
    "em_santa":   {"name": "Santa (ES Male)",          "language": "es",    "lang_code": "e"},
    # ── French (lang_code='f') ────────────────────────────────────────────
    "ff_siwis":   {"name": "Siwis ★ (FR Female)",      "language": "fr",    "lang_code": "f"},  # B-
    # ── Italian (lang_code='i') ───────────────────────────────────────────
    "if_sara":    {"name": "Sara (IT Female)",         "language": "it",    "lang_code": "i"},  # C
    "im_nicola":  {"name": "Nicola (IT Male)",         "language": "it",    "lang_code": "i"},  # C
    # ── Brazilian Portuguese (lang_code='p') ──────────────────────────────
    "pf_dora":    {"name": "Dora (PT-BR Female)",      "language": "pt-br", "lang_code": "p"},
    "pm_alex":    {"name": "Alex (PT-BR Male)",        "language": "pt-br", "lang_code": "p"},
    "pm_santa":   {"name": "Santa (PT-BR Male)",       "language": "pt-br", "lang_code": "p"},
    # ── Japanese (lang_code='j') — requires: pip install misaki[ja] ───────
    "jf_alpha":   {"name": "Alpha (JA Female)",        "language": "ja",    "lang_code": "j"},  # C+
    "jf_gongitsune": {"name": "Gongitsune (JA Female)","language": "ja",   "lang_code": "j"},  # C
    "jf_nezumi":  {"name": "Nezumi (JA Female)",       "language": "ja",    "lang_code": "j"},  # C-
    "jf_tebukuro":{"name": "Tebukuro (JA Female)",     "language": "ja",    "lang_code": "j"},  # C
    "jm_kumo":    {"name": "Kumo (JA Male)",           "language": "ja",    "lang_code": "j"},  # C-
    # ── Mandarin Chinese (lang_code='z') — requires: pip install misaki[zh]
    "zf_xiaobei": {"name": "Xiaobei (ZH Female)",      "language": "zh",    "lang_code": "z"},  # D
    "zf_xiaoni":  {"name": "Xiaoni (ZH Female)",       "language": "zh",    "lang_code": "z"},  # D
    "zf_xiaoxiao":{"name": "Xiaoxiao (ZH Female)",     "language": "zh",    "lang_code": "z"},  # D
    "zf_xiaoyi":  {"name": "Xiaoyi (ZH Female)",       "language": "zh",    "lang_code": "z"},  # D
    "zm_yunjian": {"name": "Yunjian (ZH Male)",        "language": "zh",    "lang_code": "z"},  # D
    "zm_yunxi":   {"name": "Yunxi (ZH Male)",          "language": "zh",    "lang_code": "z"},  # D
    "zm_yunxia":  {"name": "Yunxia (ZH Male)",         "language": "zh",    "lang_code": "z"},  # D
    "zm_yunyang": {"name": "Yunyang (ZH Male)",        "language": "zh",    "lang_code": "z"},  # D
}


class KokoroEngine(TTSEngine):
    """Lazily loads KPipeline instances keyed by lang_code."""

    def __init__(self) -> None:
        self._pipelines: dict[str, object] = {}
        self._ready = False
        self._load()

    def _load(self) -> None:
        try:
            from kokoro import KPipeline  # noqa: F401 — validate import at startup
            self._KPipeline = KPipeline
            self._ready = True
            logger.info("Kokoro engine initialized (models load on first synthesis call)")
        except ImportError as exc:
            logger.error("Kokoro package not installed: %s", exc)

    def _get_pipeline(self, lang_code: str):
        if lang_code not in self._pipelines:
            logger.info("Loading Kokoro pipeline for lang_code='%s'", lang_code)
            self._pipelines[lang_code] = self._KPipeline(
                lang_code=lang_code,
                repo_id=_KOKORO_REPO_ID,
            )
        return self._pipelines[lang_code]

    def synthesize(
        self,
        text: str,
        voice: str,
        language: str = "en-us",
        speed: float = 1.0,
        **kwargs,
    ) -> bytes:
        if not self._ready:
            raise RuntimeError("Kokoro engine failed to initialize. Check installation.")

        voice_meta = _VOICES.get(voice)
        if voice_meta is None:
            raise ValueError(f"Unknown Kokoro voice: {voice}")

        lang_code = voice_meta["lang_code"]
        pipeline = self._get_pipeline(lang_code)

        chunks: list[np.ndarray] = []
        # KPipeline returns a generator of (graphemes, phonemes, audio_array) tuples
        for _, _, audio in pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+"):
            if audio is not None and len(audio) > 0:
                chunks.append(np.asarray(audio, dtype=np.float32))

        if not chunks:
            raise RuntimeError("Kokoro produced no audio output.")

        combined = np.concatenate(chunks)
        processed, sr = postprocess(combined, src_rate=24000)
        return audio_to_bytes(processed, sr)

    def list_voices(self) -> list[dict]:
        return [
            {"id": vid, "name": meta["name"], "language": meta["language"], "engine": "kokoro"}
            for vid, meta in _VOICES.items()
        ]

    def is_ready(self) -> bool:
        return self._ready
