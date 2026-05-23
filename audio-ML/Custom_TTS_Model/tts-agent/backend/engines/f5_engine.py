"""F5-TTS engine wrapper for zero-shot voice cloning (MIT-licensed — commercial-safe).

Language support — multi-model architecture
--------------------------------------------
F5TTS_v1_Base was trained primarily on English and Mandarin Chinese.  For other
languages (Hindi, Japanese, Arabic, Korean, etc.) the model's character tokeniser
lacks vocabulary coverage, causing it to produce garbled or wrong-language output.
An additional problem specific to Hindi is that F5-TTS internally runs OpenAI
Whisper ASR on the reference audio to auto-transcribe it.  Whisper frequently
mis-classifies Hindi speech as Urdu and outputs Perso-Arabic script, which then
conditions the synthesiser to produce Arabic/Urdu-sounding speech — even when a
correct Devanagari ref_text is supplied by the caller.

Two complementary fixes are applied:

  1. Language-specific model routing: when a language override is configured
     (e.g. SPRINGLab/F5-Hindi-24KHz for Hindi), the engine loads and uses that
     checkpoint instead of the base model.  Language-specific models are trained
     on Indic data, use Devanagari in their vocabulary, and their internal Whisper
     is tuned to correctly identify the target language — eliminating the Hindi→Urdu
     drift at the source.

  2. IPA phonemisation fallback: when no language-specific model is available
     (or when it is disabled via settings), the engine converts non-Latin text to
     an ASCII IPA approximation so that the base model's tokeniser can handle it.
     This does NOT fix the Whisper mis-classification, but produces better output
     than raw Devanagari on the base model.

Logging
-------
Every synthesis call emits structured INFO lines at key pipeline stages:

  [F5] request  | voice=clone:speaker | lang=hi | script=Devanagari | chars=312
  [F5] model    | selected=SPRINGLab/F5-Hindi-24KHz | native_script=True | reason=hi override
  [F5] phonemise| lang=hi → espeak=hi | original='जीवन...' | ascii='jiivan...'
  [F5] synthesis| gen_text='जीवन...' | phonemised=False | model=SPRINGLab/F5-Hindi-24KHz
  [F5] output   | voice=clone:speaker | lang=hi | duration=12.34s | sr=24000
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np

from backend.engines.base import TTSEngine
from backend.utils.audio import postprocess, audio_to_bytes, load_audio
from backend.utils.logger import get_logger
from configs.settings import settings

logger = get_logger(__name__)


# ── IPA → ASCII approximation table ──────────────────────────────────────────
#
# Built from the IPA characters that espeak-ng emits for South/East Asian and
# other non-Latin-script languages.  All ASCII characters pass through unchanged.
# 'ː' (U+02D0 length mark) is handled procedurally to double the preceding vowel.
# Combining tilde (U+0303) marks nasalisation → rendered as 'n'.
#
_IPA_TO_ASCII: dict[str, str] = {
    # Vowels
    'ə': 'a',    # U+0259 schwa
    'ɛ': 'e',    # U+025B open-mid front
    'ɪ': 'i',    # U+026A near-close front
    'ɔ': 'o',    # U+0254 open-mid back rounded
    'ʌ': 'a',    # U+028C open-mid back unrounded (Hindi अ short)
    'ɑ': 'aa',   # U+0251 open back unrounded
    'æ': 'ae',   # U+00E6 near-open front
    'ʊ': 'u',    # U+028A near-close back
    # Consonants
    'ɟ': 'j',    # U+025F palatal stop (Hindi ज)
    'ɡ': 'g',    # U+0261 voiced velar stop
    'ɾ': 'r',    # U+027E alveolar flap (Hindi र)
    'ɽ': 'r',    # U+027D retroflex flap
    'ʂ': 'sh',   # U+0282 retroflex fricative (Hindi ष)
    'ʃ': 'sh',   # U+0283 postalveolar fricative (Hindi श)
    'ʒ': 'zh',   # U+0292 voiced postalveolar
    'ʋ': 'v',    # U+028B labiodental approx (Hindi व)
    'ŋ': 'ng',   # U+014B velar nasal
    'ɲ': 'n',    # U+0272 palatal nasal
    'ɳ': 'n',    # U+0273 retroflex nasal (Hindi ण)
    'ʈ': 't',    # U+0288 retroflex voiceless stop (Hindi ट)
    'ɖ': 'd',    # U+0256 retroflex voiced stop (Hindi ड)
    'ɦ': 'h',    # U+0266 breathy h
    'ɹ': 'r',    # U+0279 alveolar approximant
    'ɫ': 'l',    # U+026B dark l
    'χ': 'kh',   # U+03C7 voiceless uvular fricative
    'ʁ': 'r',    # U+0281 voiced uvular fricative
    'ħ': 'h',    # voiceless pharyngeal
    'ʔ': '',     # U+0294 glottal stop → remove
    'ɐ': 'a',    # U+0250 near-open central
    # Suprasegmentals (remove — not phonemic for F5-TTS)
    'ˈ': '',     # U+02C8 primary stress
    'ˌ': '',     # U+02CC secondary stress
    'ˑ': '',     # U+02D1 half-length
    '͡': '',     # U+0361 tie bar (affricate ligature)
    # Modifier letters
    'ʰ': 'h',    # U+02B0 aspiration (bh th dh kh ph)
    'ʷ': 'w',    # U+02B7 labialization
    'ʲ': 'y',    # U+02B2 palatalisation
    # Combining diacritics
    '̃': 'n',  # combining tilde → nasalisation
    '̪': '',   # combining dental
    '̚': '',   # no audible release
    '̤': '',   # breathy voice diacritic
}

# Precomposed nasal-vowel characters (Unicode NFC)
_NASAL_VOWELS: dict[str, str] = {
    'ã': 'an', 'ẽ': 'en', 'ĩ': 'in', 'õ': 'on', 'ũ': 'un',
}

# ── Unicode script ranges for detection ──────────────────────────────────────

_SCRIPT_RANGES: list[tuple[int, int, str]] = [
    (0x0900, 0x097F, 'Devanagari'),    # Hindi, Marathi, Sanskrit
    (0x0980, 0x09FF, 'Bengali'),
    (0x0A00, 0x0A7F, 'Gurmukhi'),
    (0x0A80, 0x0AFF, 'Gujarati'),
    (0x0B00, 0x0B7F, 'Odia'),
    (0x0B80, 0x0BFF, 'Tamil'),
    (0x0C00, 0x0C7F, 'Telugu'),
    (0x0C80, 0x0CFF, 'Kannada'),
    (0x0D00, 0x0D7F, 'Malayalam'),
    (0x0600, 0x06FF, 'Arabic'),
    (0x0400, 0x04FF, 'Cyrillic'),
    (0x0E00, 0x0E7F, 'Thai'),
    (0x3040, 0x309F, 'Hiragana'),
    (0x30A0, 0x30FF, 'Katakana'),
    (0x4E00, 0x9FFF, 'CJK'),
    (0xAC00, 0xD7FF, 'Hangul'),
]

# espeak-ng language codes keyed by BCP-47 prefix
_ESPEAK_LANG: dict[str, str] = {
    'hi': 'hi', 'mr': 'mr', 'bn': 'bn', 'gu': 'gu',
    'pa': 'pa', 'ta': 'ta', 'te': 'te', 'kn': 'kn', 'ml': 'ml',
    'ar': 'ar', 'ru': 'ru', 'el': 'el', 'th': 'th',
    'ko': 'ko', 'ja': 'ja', 'zh': 'cmn',
}

# ── Language-specific model routing ──────────────────────────────────────────
#
# Each entry in _LANG_MODEL_OVERRIDES describes one language-specific model:
#
#   repo_id        — HuggingFace repo ID, or a built-in config name (no '/')
#   ckpt_file      — specific checkpoint path within the repo ('' = auto-detect)
#   vocab_file     — specific vocab path within the repo ('' = auto-detect)
#   is_native_script — True → pass text in its native script (Devanagari, kana,
#                      …); False → phonemise to ASCII first via espeak-ng
#
# Built once from settings so operators can override via environment variables.
#
class _LangModelConfig:
    __slots__ = ('repo_id', 'ckpt_file', 'vocab_file', 'arch_model', 'is_native_script')

    def __init__(
        self,
        repo_id: str,
        ckpt_file: str = '',
        vocab_file: str = '',
        arch_model: str = '',       # F5-TTS architecture config (e.g. "F5TTS_Small")
                                    # empty → use settings.F5_MODEL_NAME (F5TTS_v1_Base)
        is_native_script: bool = True,
    ) -> None:
        self.repo_id = repo_id
        self.ckpt_file = ckpt_file
        self.vocab_file = vocab_file
        self.arch_model = arch_model
        self.is_native_script = is_native_script

    def __repr__(self) -> str:
        arch = self.arch_model or settings.F5_MODEL_NAME
        return (
            f"<LangModel repo={self.repo_id!r} arch={arch!r} "
            f"ckpt={self.ckpt_file or '(auto)'!r} "
            f"vocab={self.vocab_file or '(auto)'!r} native={self.is_native_script}>"
        )


def _build_lang_model_map() -> dict[str, _LangModelConfig]:
    overrides: dict[str, _LangModelConfig] = {}

    hi = settings.F5_HINDI_MODEL_NAME.strip()
    if hi:
        overrides['hi'] = _LangModelConfig(
            repo_id=hi,
            ckpt_file=settings.F5_HINDI_CKPT_FILE.strip(),
            vocab_file=settings.F5_HINDI_VOCAB_FILE.strip(),
            is_native_script=True,   # Devanagari vocab
        )

    ja = settings.F5_JAPANESE_MODEL_NAME.strip()
    if ja:
        overrides['ja'] = _LangModelConfig(
            repo_id=ja,
            ckpt_file=settings.F5_JAPANESE_CKPT_FILE.strip(),
            vocab_file=settings.F5_JAPANESE_VOCAB_FILE.strip(),
            arch_model=settings.F5_JAPANESE_ARCH_MODEL.strip(),
            is_native_script=True,   # Hiragana + Katakana + Kanji vocab
        )

    return overrides


# Populated in F5Engine._load() once f5-tts is confirmed importable.
_LANG_MODEL_OVERRIDES: dict[str, _LangModelConfig] = {}


def detect_script(text: str) -> str:
    """Return the dominant non-Latin script name in *text*, or 'Latin'."""
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for lo, hi, name in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
    return max(counts, key=counts.__getitem__) if counts else 'Latin'


def _has_non_latin(text: str) -> bool:
    """True if text contains letters outside ASCII/Latin-Extended range."""
    return any(ord(ch) > 0x024F and ch.isalpha() for ch in text)


def _ipa_to_ascii(ipa: str) -> str:
    """
    Convert an IPA string (espeak-ng output) to an ASCII approximation
    that F5-TTS can tokenise.

    Key operations:
    - 'ː' (length mark) doubles the preceding vowel (a→aa, i→ii, etc.)
    - Precomposed nasal vowels (ẽ, ã, …) → two-char ASCII code
    - All other IPA chars mapped via _IPA_TO_ASCII; unmapped chars dropped
    """
    result: list[str] = []
    for ch in ipa:
        # Precomposed nasal vowels
        if ch in _NASAL_VOWELS:
            result.append(_NASAL_VOWELS[ch])
            continue
        # Length mark: duplicate preceding vowel
        if ch == 'ː':
            if result and result[-1] in 'aeiou':
                result.append(result[-1])
            continue
        # ASCII passes through
        if ord(ch) < 128:
            result.append(ch)
            continue
        # Lookup in map
        if ch in _IPA_TO_ASCII:
            result.append(_IPA_TO_ASCII[ch])
            continue
        # Try NFD decomposition (e.g. é → e + combining acute)
        decomposed = unicodedata.normalize('NFD', ch)
        if len(decomposed) > 1:
            for sub in decomposed:
                if ord(sub) < 128:
                    result.append(sub)
                elif sub in _IPA_TO_ASCII:
                    result.append(_IPA_TO_ASCII[sub])
        # Unmapped non-ASCII: silently skip

    return re.sub(r'\s+', ' ', ''.join(result)).strip()


def _find_espeak_library() -> tuple[str | None, str | None]:
    """Return (library_path, data_path) for a usable espeak-ng installation."""
    lib_candidates = [
        '/opt/homebrew/lib/libespeak-ng.dylib',
        '/usr/local/lib/libespeak-ng.dylib',
        '/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1',
        '/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1',
        '/usr/local/lib/libespeak-ng.so.1',
    ]
    data_candidates = [
        '/opt/homebrew/share/espeak-ng-data',
        '/usr/local/share/espeak-ng-data',
        '/usr/share/espeak-ng-data',
        '/usr/lib/espeak-ng-data',
    ]
    lib_path = next((p for p in lib_candidates if Path(p).exists()), None)
    data_path = next((p for p in data_candidates if Path(p).is_dir()), None)

    if lib_path is None or data_path is None:
        try:
            import espeakng_loader as _el
            return _el.get_library_path(), _el.get_data_path()
        except Exception:
            pass
    return lib_path, data_path


def phonemise_text(text: str, language: str) -> tuple[str, bool]:
    """
    Convert non-Latin-script text to an ASCII phoneme approximation for F5-TTS.

    Returns (processed_text, was_phonemised).
    Falls back to raw text if phonemisation is not available.
    """
    if not _has_non_latin(text):
        return text, False

    lang_prefix = language.split('-')[0].lower()
    espeak_lang = _ESPEAK_LANG.get(lang_prefix)

    if not espeak_lang:
        logger.warning(
            "[F5] phonemise | no espeak mapping for language='%s' — "
            "sending raw text to F5-TTS (output may be incorrect)", language
        )
        return text, False

    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        from phonemizer.backend import EspeakBackend
        from phonemizer.separator import Separator

        lib_path, data_path = _find_espeak_library()
        if lib_path:
            EspeakWrapper.set_library(lib_path)
        if data_path:
            EspeakWrapper.set_data_path(data_path)

        backend = EspeakBackend(
            espeak_lang,
            with_stress=False,
            language_switch='remove-flags',
            words_mismatch='ignore',
        )
        sep = Separator(phone='', word=' ', syllable='')
        ipa_raw = backend.phonemize([text], separator=sep)[0]
        ascii_text = _ipa_to_ascii(ipa_raw)

        if not ascii_text:
            logger.warning(
                "[F5] phonemise | empty result for lang=%s — using raw text", language
            )
            return text, False

        _trunc = lambda s, n: (s[:n] + '…') if len(s) > n else s
        logger.info(
            "[F5] phonemise | lang=%s → espeak=%s | original=%r | ascii=%r",
            language, espeak_lang, _trunc(text, 60), _trunc(ascii_text, 60),
        )
        return ascii_text, True

    except Exception as exc:
        logger.error(
            "[F5] phonemise | FAILED for lang=%s: %s — using raw text", language, exc
        )
        return text, False


# ── Engine ────────────────────────────────────────────────────────────────────

class F5Engine(TTSEngine):
    """Wraps F5-TTS for zero-shot voice cloning from user-supplied reference audio.

    Model selection strategy
    ------------------------
    - Language-specific overrides (e.g. SPRINGLab/F5-Hindi-24KHz for 'hi') are
      loaded on first use and cached.  These models understand their target script
      natively, so IPA phonemisation is skipped for them.
    - The base model (F5TTS_v1_Base) is used for all other languages; non-Latin
      text is phonemised to ASCII via espeak-ng before synthesis.
    - All models are loaded lazily on first synthesis call (not at server start)
      to keep startup fast.
    """

    def __init__(self) -> None:
        self._F5TTS = None               # F5TTS class (set in _load)
        self._models: dict[str, object] = {}  # model name → F5TTS instance
        self._ready = False
        self._load()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _load(self) -> None:
        global _LANG_MODEL_OVERRIDES
        try:
            from f5_tts.api import F5TTS
            self._F5TTS = F5TTS
            self._ready = True
            _LANG_MODEL_OVERRIDES = _build_lang_model_map()
            logger.info(
                "F5-TTS engine initialized | base=%s | lang_overrides=%s",
                settings.F5_MODEL_NAME,
                {k: v for k, v in _LANG_MODEL_OVERRIDES.items()} or "none",
            )
        except ImportError as exc:
            logger.error("f5-tts package not installed: %s", exc)

    # ── Model selection helpers ───────────────────────────────────────────────

    def _config_for_language(self, language: str) -> tuple[_LangModelConfig | None, str]:
        """Return (lang_config, display_reason) for the given BCP-47 tag.

        Returns (None, reason) when no language-specific model is configured;
        the caller should fall back to the base model + phonemisation.
        """
        lang_prefix = language.split('-')[0].lower()
        cfg = _LANG_MODEL_OVERRIDES.get(lang_prefix)
        if cfg:
            return cfg, f"{lang_prefix} override → {cfg.repo_id}"
        return None, "base model (no language override)"

    def _get_model(self, cfg: '_LangModelConfig | None'):
        """Lazy-load and cache an F5TTS instance for the given config.

        *cfg=None* means the base built-in model (F5TTS_v1_Base).
        HuggingFace repos (repo_id contains '/') are downloaded on first use
        and cached in the standard HF cache directory.
        """
        cache_key = cfg.repo_id if cfg else settings.F5_MODEL_NAME
        if cache_key not in self._models:
            if cfg is None:
                logger.info("Loading F5-TTS built-in model '%s' …", settings.F5_MODEL_NAME)
                self._models[cache_key] = self._F5TTS(model=settings.F5_MODEL_NAME)
                logger.info("F5-TTS model '%s' loaded successfully.", settings.F5_MODEL_NAME)
            elif '/' in cfg.repo_id:
                self._models[cache_key] = self._load_hf_model(cfg)
            else:
                # Built-in config name supplied as an override (unusual)
                logger.info("Loading F5-TTS built-in model '%s' …", cfg.repo_id)
                self._models[cache_key] = self._F5TTS(model=cfg.repo_id)
                logger.info("F5-TTS model '%s' loaded successfully.", cfg.repo_id)
        return self._models[cache_key]

    def _load_hf_model(self, cfg: '_LangModelConfig'):
        """Download a HuggingFace F5-TTS checkpoint and instantiate it.

        If cfg.ckpt_file / cfg.vocab_file are specified, those exact paths are
        downloaded (supports checkpoints in subdirectories, e.g. Jmica/F5TTS).
        Otherwise, the first .safetensors (or .pt) and vocab.txt are auto-detected
        in the repo root.

        Files are cached in HuggingFace's standard cache directory — subsequent
        loads are instant (no re-download).
        """
        from huggingface_hub import hf_hub_download, list_repo_files

        repo_id = cfg.repo_id
        logger.info("Resolving HuggingFace model '%s' …", repo_id)

        # ── Checkpoint file ───────────────────────────────────────────────
        if cfg.ckpt_file:
            ckpt_filename = cfg.ckpt_file
            logger.info("Using configured checkpoint: %s", ckpt_filename)
        else:
            # Auto-detect: prefer .safetensors over .pt in repo root
            try:
                all_files = list(list_repo_files(repo_id))
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot list files in HuggingFace repo '{repo_id}': {exc}"
                ) from exc
            candidates = [f for f in all_files if f.endswith('.safetensors')]
            if not candidates:
                candidates = [f for f in all_files if f.endswith('.pt')]
            if not candidates:
                raise RuntimeError(
                    f"No .safetensors or .pt checkpoint found in '{repo_id}'. "
                    f"Files: {all_files}"
                )
            ckpt_filename = candidates[0]
            logger.info("Auto-detected checkpoint: %s", ckpt_filename)

        logger.info(
            "Downloading '%s' from '%s' (cached after first run)…",
            ckpt_filename, repo_id,
        )
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_filename)
        logger.info("Checkpoint cached at: %s", ckpt_path)

        # ── Vocab file ────────────────────────────────────────────────────
        vocab_path = ''
        if cfg.vocab_file:
            vocab_path = hf_hub_download(repo_id=repo_id, filename=cfg.vocab_file)
            logger.info("Vocab cached at: %s", vocab_path)
        else:
            # Try to auto-detect vocab.txt in repo root
            try:
                if not cfg.ckpt_file:   # all_files already fetched above
                    pass
                else:
                    all_files = list(list_repo_files(repo_id))
                if 'vocab.txt' in all_files:
                    vocab_path = hf_hub_download(repo_id=repo_id, filename='vocab.txt')
                    logger.info("Auto-detected vocab at: %s", vocab_path)
            except Exception:
                pass  # non-fatal; F5TTS will use its default vocab

        # ── Instantiate ───────────────────────────────────────────────────
        # Use the architecture config from the lang override (e.g. "F5TTS_Small"
        # for the Jmica Japanese checkpoint which has depth=18, not 22).
        # Falls back to settings.F5_MODEL_NAME (F5TTS_v1_Base) when not set.
        arch = cfg.arch_model or settings.F5_MODEL_NAME
        logger.info(
            "Loading F5-TTS | arch=%s | ckpt=%s | vocab=%s",
            arch, ckpt_path, vocab_path or "(default)",
        )
        instance = self._F5TTS(
            model=arch,
            ckpt_file=ckpt_path,
            vocab_file=vocab_path,
        )
        logger.info("HuggingFace model '%s' loaded successfully.", repo_id)
        return instance

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        voice: str,
        language: str = "en-us",
        speed: float = 1.0,
        reference_audio: str | None = None,
        reference_text: str | None = None,
        **kwargs,
    ) -> bytes:
        if not self._ready:
            raise RuntimeError("F5-TTS engine failed to initialize. Check installation.")

        if not reference_audio:
            raise ValueError("F5-TTS requires a reference_audio path for voice cloning.")

        ref_path = settings.voice_samples_dir / reference_audio
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_path}")

        # ── Script detection & model selection ───────────────────────────
        script = detect_script(text)
        lang_cfg, reason = self._config_for_language(language)
        is_native_script = lang_cfg.is_native_script if lang_cfg else False
        model_display = lang_cfg.repo_id if lang_cfg else settings.F5_MODEL_NAME
        _trunc = lambda s, n: (s[:n] + '…') if len(s) > n else s

        logger.info(
            "[F5] request | voice=%s | lang=%s | script=%s | chars=%d | "
            "speed=%.2f | ref_audio=%s | ref_text_chars=%d",
            voice, language, script, len(text), speed,
            reference_audio, len(reference_text or ''),
        )
        logger.info(
            "[F5] model | selected=%s | native_script=%s | reason=%s",
            model_display, is_native_script, reason,
        )

        # ── Text preparation ─────────────────────────────────────────────
        #
        # Native-script models (Hindi 24KHz, Jmica Japanese, …) have the target
        # script in their vocabulary and their internal Whisper is language-tuned.
        # Pass text in its native form — phonemising to ASCII would discard the
        # script information the model was trained to use.
        #
        # Base model (English + Mandarin only): phonemise non-Latin text to ASCII
        # so the tokeniser can handle it.  This is a partial fix; it does not
        # prevent Whisper from mis-identifying the reference audio language.
        #
        if is_native_script:
            gen_text = text
            was_phonemised = False
            ref_text_out = reference_text
            if reference_text:
                logger.info(
                    "[F5] ref_text | mode=native | ref=%r",
                    _trunc(reference_text, 60),
                )
        else:
            gen_text, was_phonemised = phonemise_text(text, language)

            ref_text_out = reference_text
            if reference_text and _has_non_latin(reference_text):
                ref_text_out, _ = phonemise_text(reference_text, language)
                logger.info(
                    "[F5] ref_text | mode=phonemised | original=%r | ascii=%r",
                    _trunc(reference_text, 40), _trunc(ref_text_out or '', 40),
                )

        if not gen_text.strip():
            raise ValueError("Text preparation produced empty string — cannot synthesise.")

        # Warn loudly when ref_text is absent for non-English: F5-TTS internally
        # runs Whisper ASR which may mis-detect the language (Hindi → Urdu, etc.)
        lang_prefix = language.split('-')[0].lower()
        if not ref_text_out and lang_prefix not in ('en', 'zh'):
            logger.warning(
                "[F5] ref_text | MISSING for lang=%s — F5-TTS will run Whisper "
                "ASR on the reference audio which may mis-identify the language "
                "(e.g. Hindi→Urdu). Always provide a reference transcript.",
                lang_prefix,
            )

        # ── Synthesis ────────────────────────────────────────────────────
        model = self._get_model(lang_cfg)
        logger.info(
            "[F5] synthesis | voice=%s | lang=%s | script=%s | model=%s | "
            "phonemised=%s | gen_text=%r",
            voice, language, script, model_display, was_phonemised,
            _trunc(gen_text, 80),
        )

        wav, sr, _ = model.infer(
            ref_file=str(ref_path),
            ref_text=ref_text_out or "",
            gen_text=gen_text,
            speed=speed,
        )

        audio = np.asarray(wav, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()

        processed, out_sr = postprocess(audio, src_rate=sr)
        duration = len(processed) / out_sr if len(processed) > 0 else 0.0

        logger.info(
            "[F5] output | voice=%s | lang=%s | script=%s | model=%s | "
            "duration=%.2fs | sample_rate=%d | phonemised=%s",
            voice, language, script, model_display, duration, out_sr, was_phonemised,
        )

        return audio_to_bytes(processed, out_sr)

    def list_voices(self) -> list[dict]:
        """Return voice entries built from voice_samples/ directory."""
        samples_dir = settings.voice_samples_dir
        voices = []
        for p in sorted(samples_dir.iterdir()):
            if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}:
                voices.append({
                    "id": f"clone:{p.stem}",
                    "name": f"{p.stem} (Cloned)",
                    "language": "any",
                    "engine": "f5-tts",
                    "reference_file": p.name,
                })
        return voices

    def is_ready(self) -> bool:
        return self._ready
