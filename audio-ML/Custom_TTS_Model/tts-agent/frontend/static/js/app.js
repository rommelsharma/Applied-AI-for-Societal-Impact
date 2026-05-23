/* TTS Agent — frontend controller */

const $ = id => document.getElementById(id);

const SSML_EXAMPLE = `Welcome to the documentary narration.<pause ms="500"/>

Today we explore <emphasis level="strong">climate change</emphasis>
and its far-reaching consequences for life on Earth.<break strength="paragraph"/>

The <say-as interpret-as="characters">IPCC</say-as> reports that global
temperatures have risen by <emphasis level="moderate">1.1 degrees Celsius</emphasis>
since pre-industrial times.<pause ms="300"/>

Yet there is reason for <emphasis level="moderate">cautious optimism</emphasis>.
Renewable energy capacity has grown fivefold in a single decade.<pause ms="400"/>
The question now is whether the transition will happen fast enough.`;

// BCP-47 language tags for clone mode, keyed by dropdown value
const LANGUAGE_TAGS = {
  en:  'en-us',
  hi:  'hi',
  ja:  'ja',
  zh:  'zh',
  es:  'es',
  fr:  'fr',
  de:  'de',
  it:  'it',
  ko:  'ko',
  pt:  'pt-br',
  ar:  'ar',
  ru:  'ru',
};

const state = {
  mode: 'builtin',        // 'builtin' | 'clone'
  inputFormat: 'plain',   // 'plain' | 'ssml'
  voices: [],
  samples: [],
  activeSample: null,     // filename of the validated sample for synthesis
  samplePreviewUrl: null, // blob URL for the preview player
};

// ── Initialisation ──────────────────────────────────────────────────────────

async function init() {
  await Promise.all([fetchHealth(), fetchVoices(), fetchSamples()]);
  bindEvents();
  updateSynthBtn();
}

async function fetchHealth() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    const badge = $('device-badge');
    badge.textContent = `device: ${data.device}`;
    badge.className = `badge ${data.device}`;
  } catch { /* non-blocking */ }
}

async function fetchVoices() {
  try {
    const res = await fetch('/voices');
    state.voices = await res.json();
    populateVoiceSelect();
  } catch (e) {
    showError('Failed to load voices: ' + e.message);
  }
}

async function fetchSamples() {
  try {
    const res = await fetch('/voice-samples');
    state.samples = await res.json();
    // Keep the hidden select in sync (used as state carrier for synthesis)
    populateSampleSelect();
  } catch { /* non-critical */ }
}

// ── Populate selects ────────────────────────────────────────────────────────

function populateVoiceSelect() {
  const sel = $('voice-select');
  const builtinVoices = state.voices.filter(v => v.engine === 'kokoro');
  sel.innerHTML = builtinVoices
    .map(v => `<option value="${v.id}">${v.name} — ${v.language.toUpperCase()}</option>`)
    .join('');
}

function populateSampleSelect() {
  // Hidden select kept in sync so the synthesize path can read a value
  const sel = $('sample-select');
  const fromFiles = state.samples.map(s => {
    const dur = s.duration_seconds != null ? ` (${s.duration_seconds.toFixed(1)}s)` : '';
    const warn = s.duration_seconds != null && (s.duration_seconds < 3 || s.duration_seconds > 15) ? ' ⚠' : '';
    return { filename: s.filename, label: `${s.filename}${dur}${warn}` };
  });
  sel.innerHTML = '<option value="">—</option>' +
    fromFiles.map(f => `<option value="${f.filename}">${f.label}</option>`).join('');

  // Re-select active sample if it's still in the list
  if (state.activeSample) {
    for (let i = 0; i < sel.options.length; i++) {
      if (sel.options[i].value === state.activeSample) {
        sel.selectedIndex = i;
        return;
      }
    }
    // Active sample no longer in list — clear it
    state.activeSample = null;
    updateSynthBtn();
  }
}

// ── Events ───────────────────────────────────────────────────────────────────

function bindEvents() {
  // Tab switching
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      state.mode = tab.dataset.mode;
      $(`panel-${state.mode}`).classList.add('active');
      updateSynthBtn();
    });
  });

  // Input format toggle
  $('input-format').addEventListener('change', () => {
    state.inputFormat = $('input-format').value;
    const isSsml = state.inputFormat === 'ssml';
    $('ssml-hint').classList.toggle('hidden', !isSsml);
    $('ssml-example-btn').classList.toggle('hidden', !isSsml);
    $('text-input').placeholder = isSsml
      ? 'Enter SSML-tagged text, e.g.  Hello <pause ms="300"/> world.'
      : 'Enter the text you want to convert to speech…';
  });

  // SSML example button
  $('ssml-example-btn').addEventListener('click', () => {
    $('text-input').value = SSML_EXAMPLE;
    $('char-count').textContent = `${SSML_EXAMPLE.length} / 5000`;
    updateSynthBtn();
  });

  // Char count
  $('text-input').addEventListener('input', () => {
    const len = $('text-input').value.length;
    $('char-count').textContent = `${len} / 5000`;
    updateSynthBtn();
  });

  // Speed slider
  $('speed-slider').addEventListener('input', () => {
    $('speed-val').textContent = parseFloat($('speed-slider').value).toFixed(1) + '×';
  });

  // Clone language dropdown
  $('clone-language').addEventListener('change', updateSynthBtn);

  // Upload file input — reset preview & active sample when a new file is chosen
  $('upload-file').addEventListener('change', () => {
    hideSamplePreview();
    $('upload-status').textContent = '';
    // Don't clear activeSample here — user may have just browsed; wait for upload
  });

  // Upload
  $('upload-btn').addEventListener('click', handleUpload);

  // Synthesize
  $('synthesize-btn').addEventListener('click', handleSynthesize);
}

function updateSynthBtn() {
  const hasText = $('text-input').value.trim().length > 0;
  let hasVoice;
  if (state.mode === 'builtin') {
    hasVoice = $('voice-select').value !== '';
  } else {
    // Clone mode requires a validated (uploaded) sample
    hasVoice = !!state.activeSample;
  }
  $('synthesize-btn').disabled = !(hasText && hasVoice);

  // Show a nudge if in clone mode and no sample is active yet
  if (state.mode === 'clone' && !state.activeSample) {
    $('synthesize-btn').title = 'Upload and verify a voice sample first (Step 1 above)';
  } else {
    $('synthesize-btn').title = '';
  }
}

// ── Upload ───────────────────────────────────────────────────────────────────

async function handleUpload() {
  const file = $('upload-file').files[0];
  if (!file) { showError('Select a file first.'); return; }

  const form = new FormData();
  form.append('file', file);

  hideSamplePreview();
  $('upload-status').textContent = 'Uploading…';
  hideError();

  try {
    const res = await fetch('/upload-voice-sample', { method: 'POST', body: form });
    if (!res.ok) throw new Error((await res.json()).detail);
    const data = await res.json();
    $('upload-status').textContent = '✓ Uploaded';
    if (data.warning) showError('Warning: ' + data.warning);

    // Show audio preview from the local file object (instant, no extra round-trip)
    showSamplePreview(file);

    // Update active sample — this is now the voice used for synthesis
    state.activeSample = data.filename;

    // Keep hidden select in sync
    await fetchSamples();
    populateSampleSelect();
    updateSynthBtn();
  } catch (e) {
    $('upload-status').textContent = '';
    showError('Upload failed: ' + e.message);
  }
}

function showSamplePreview(file) {
  if (state.samplePreviewUrl) {
    URL.revokeObjectURL(state.samplePreviewUrl);
  }
  state.samplePreviewUrl = URL.createObjectURL(file);
  const player = $('sample-player');
  player.src = state.samplePreviewUrl;
  player.load();
  $('sample-preview').classList.remove('hidden');
}

function hideSamplePreview() {
  $('sample-preview').classList.add('hidden');
  const player = $('sample-player');
  player.pause();
  player.src = '';
  if (state.samplePreviewUrl) {
    URL.revokeObjectURL(state.samplePreviewUrl);
    state.samplePreviewUrl = null;
  }
}

// ── Synthesize ───────────────────────────────────────────────────────────────

async function handleSynthesize() {
  hideError();
  showLoading(true);
  $('output-section').classList.add('hidden');

  const text = $('text-input').value.trim();
  const speed = parseFloat($('speed-slider').value);
  const use_ssml = state.inputFormat === 'ssml';

  let body;
  if (state.mode === 'builtin') {
    // Derive language from the selected voice metadata
    const selectedVoice = state.voices.find(v => v.id === $('voice-select').value);
    const language = selectedVoice ? selectedVoice.language : 'en-us';
    body = { text, voice: $('voice-select').value, language, speed, use_ssml };
  } else {
    // Clone mode — use the validated active sample
    if (!state.activeSample) {
      showError('Please upload and verify a voice sample first.');
      showLoading(false);
      return;
    }
    const langCode = $('clone-language').value;
    const language = LANGUAGE_TAGS[langCode] || 'en-us';
    const refText = $('ref-text').value.trim();

    // Warn (but don't block) when ref_text is absent for non-English languages.
    // Without it, F5-TTS runs Whisper ASR on the reference audio which can
    // mis-detect the language (e.g. Hindi speech → Urdu script → wrong output).
    if (!refText && langCode !== 'en') {
      showError(
        '⚠ Reference Transcript is empty. Without it, F5-TTS will try to ' +
        'auto-transcribe the audio and may detect the wrong language, ' +
        'producing incorrect speech. Add the transcript above for best results.'
      );
      showLoading(false);
      return;
    }

    const voiceId = `clone:${state.activeSample.replace(/\.[^.]+$/, '')}`;
    body = {
      text,
      voice: voiceId,
      language,
      speed,
      use_ssml,
      reference_audio: state.activeSample,
      reference_text: refText || null,
    };
  }

  try {
    const res = await fetch('/synthesize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    renderOutput(data);
  } catch (e) {
    showError('Synthesis failed: ' + e.message);
  } finally {
    showLoading(false);
  }
}

function renderOutput(data) {
  const player = $('audio-player');
  player.src = data.audio_url;
  player.load();

  $('duration-label').textContent = `${data.duration_seconds.toFixed(2)}s`;
  $('engine-label').textContent = `engine: ${data.engine}`;

  const link = $('download-link');
  link.href = data.audio_url;
  link.download = data.filename;

  $('output-section').classList.remove('hidden');
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function showError(msg) {
  const box = $('error-box');
  box.textContent = msg;
  box.classList.remove('hidden');
}
function hideError() { $('error-box').classList.add('hidden'); }
function showLoading(on) { $('loading').classList.toggle('hidden', !on); }

document.addEventListener('DOMContentLoaded', init);
