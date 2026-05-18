/* TTS Agent — frontend controller */

const $ = id => document.getElementById(id);

const state = {
  mode: 'builtin',       // 'builtin' | 'clone'
  voices: [],
  samples: [],
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
  const sel = $('sample-select');
  const extras = state.voices.filter(v => v.engine === 'f5-tts');
  const fromFiles = state.samples.map(s => ({
    id: `clone:${s.filename.replace(/\.[^.]+$/, '')}`,
    filename: s.filename,
    label: s.filename,
  }));
  sel.innerHTML = '<option value="">— select uploaded sample —</option>' +
    fromFiles.map(f => `<option value="${f.filename}">${f.label}</option>`).join('');
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

  // Upload
  $('upload-btn').addEventListener('click', handleUpload);

  // Synthesize
  $('synthesize-btn').addEventListener('click', handleSynthesize);
}

function updateSynthBtn() {
  const hasText = $('text-input').value.trim().length > 0;
  const hasVoice = state.mode === 'builtin'
    ? $('voice-select').value !== ''
    : $('sample-select').value !== '';
  $('synthesize-btn').disabled = !(hasText && hasVoice);
}

// ── Upload ───────────────────────────────────────────────────────────────────

async function handleUpload() {
  const file = $('upload-file').files[0];
  if (!file) { showError('Select a file first.'); return; }

  const form = new FormData();
  form.append('file', file);

  $('upload-status').textContent = 'Uploading…';
  try {
    const res = await fetch('/upload-voice-sample', { method: 'POST', body: form });
    if (!res.ok) throw new Error((await res.json()).detail);
    $('upload-status').textContent = '✓ Uploaded';
    await fetchSamples();
    populateSampleSelect();
  } catch (e) {
    $('upload-status').textContent = '';
    showError('Upload failed: ' + e.message);
  }
}

// ── Synthesize ───────────────────────────────────────────────────────────────

async function handleSynthesize() {
  hideError();
  showLoading(true);
  $('output-section').classList.add('hidden');

  const text = $('text-input').value.trim();
  const speed = parseFloat($('speed-slider').value);

  let body;
  if (state.mode === 'builtin') {
    body = { text, voice: $('voice-select').value, language: 'en-us', speed };
  } else {
    const refFile = $('sample-select').value;
    const voiceId = `clone:${refFile.replace(/\.[^.]+$/, '')}`;
    body = {
      text,
      voice: voiceId,
      language: 'en-us',
      speed,
      reference_audio: refFile,
      reference_text: $('ref-text').value.trim() || null,
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
