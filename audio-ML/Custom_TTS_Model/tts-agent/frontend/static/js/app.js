/* TTS Agent — frontend controller (v3.0 — Kokoro-82M + XTTS v2) */

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
  zh:  'zh-cn',
  es:  'es',
  fr:  'fr',
  de:  'de',
  it:  'it',
  ko:  'ko',
  pt:  'pt-br',
  ar:  'ar',
  ru:  'ru',
  pl:  'pl',
  tr:  'tr',
  nl:  'nl',
};

// ── Test suite definitions ─────────────────────────────────────────────────
const TESTS = {
  hindi: {
    referenceAudio:  'cloning-voice-clip-male-hindi-1.wav',
    referenceText:   'नमस्कार सभी को, आज मैं समय का महत्व विषय पर कुछ शब्द कहना चाहता हूँ। समय हमारे जीवन की सबसे कीमती चीज़ है।',
    synthesisText:   'जीवन में एक लक्ष्य होना ज़रूरी है। जब मन में एक सपना हो, तो हर सुबह उठने का कारण मिलता है। अपने मन से पूछो — "मैं क्या बनना चाहता हूँ?" जब यह उत्तर मिल जाए, तो उसे अपनी आत्मा में बसा लो। रास्ते में क्रोध, ईर्ष्या और निराशा आएगी। इन्हें पहचानो, इनसे लड़ो मत। अपने भीतर की आवाज़ सुनो — यही जागरूकता तुम्हें सही राह दिखाएगी।',
    language:        'hi',
  },
  japanese: {
    referenceAudio:  'cloning-voice-samples-JP.wav',
    referenceText:   '浜田山とは、いわゆる「超有名観光地」ではないけれど、東京のローカルで上質な住宅街の空気感を味わえるエリアです。',
    synthesisText:   'やばい、『ホッパーズ』まじで最高だった！メイベルがビーバーのロボットに意識を移して動物たちと一緒に生きるって設定、最初は「え、どういうこと？」ってなったけど、見てるうちにどんどん引き込まれた。キング・ジョージがかわいすぎて、ずっと笑ってたのに、最後は普通に泣いた。自然保護のメッセージがすごくリアルに伝わってきたのが良かった。',
    language:        'ja',
  },
};

const state = {
  mode:            'builtin',  // 'builtin' | 'clone' | 'tests'
  inputFormat:     'plain',
  voices:          [],
  samples:         [],
  activeSample:    null,
  samplePreviewUrl: null,
};

// ── Initialisation ──────────────────────────────────────────────────────────

async function init() {
  await Promise.all([fetchHealth(), fetchVoices(), fetchSamples()]);
  bindEvents();
  updateSynthBtn();
}

async function fetchHealth() {
  try {
    const res  = await fetch('/health');
    const data = await res.json();
    const badge = $('device-badge');
    badge.textContent = `device: ${data.device}`;
    badge.className   = `badge ${data.device}`;
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
  const fromFiles = state.samples.map(s => {
    const dur  = s.duration_seconds != null ? ` (${s.duration_seconds.toFixed(1)}s)` : '';
    const warn = s.duration_seconds != null && (s.duration_seconds < 3 || s.duration_seconds > 15) ? ' ⚠' : '';
    return { filename: s.filename, label: `${s.filename}${dur}${warn}` };
  });
  sel.innerHTML = '<option value="">—</option>' +
    fromFiles.map(f => `<option value="${f.filename}">${f.label}</option>`).join('');

  if (state.activeSample) {
    for (let i = 0; i < sel.options.length; i++) {
      if (sel.options[i].value === state.activeSample) {
        sel.selectedIndex = i;
        return;
      }
    }
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
      // Hide shared synthesis controls on Test Results tab
      const isTests = state.mode === 'tests';
      $('shared-controls').classList.toggle('hidden', isTests);
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

  $('ssml-example-btn').addEventListener('click', () => {
    $('text-input').value = SSML_EXAMPLE;
    $('char-count').textContent = `${SSML_EXAMPLE.length} / 5000`;
    updateSynthBtn();
  });

  $('text-input').addEventListener('input', () => {
    const len = $('text-input').value.length;
    $('char-count').textContent = `${len} / 5000`;
    updateSynthBtn();
  });

  $('speed-slider').addEventListener('input', () => {
    $('speed-val').textContent = parseFloat($('speed-slider').value).toFixed(1) + '×';
  });

  $('clone-language').addEventListener('change', updateSynthBtn);

  $('upload-file').addEventListener('change', () => {
    hideSamplePreview();
    $('upload-status').textContent = '';
  });

  $('upload-btn').addEventListener('click', handleUpload);
  $('synthesize-btn').addEventListener('click', handleSynthesize);

  // Test suite buttons
  $('run-all-tests-btn').addEventListener('click', () => {
    runTest('hindi');
    runTest('japanese');
  });

  document.querySelectorAll('.run-test-btn').forEach(btn => {
    btn.addEventListener('click', () => runTest(btn.dataset.test));
  });
}

function updateSynthBtn() {
  const hasText = $('text-input').value.trim().length > 0;
  let hasVoice;
  if (state.mode === 'builtin') {
    hasVoice = $('voice-select').value !== '';
  } else {
    hasVoice = !!state.activeSample;
  }
  $('synthesize-btn').disabled = !(hasText && hasVoice);

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
    if (!res.ok) {
      let detail;
      try { detail = (await res.json()).detail; } catch { detail = await res.text().catch(() => res.statusText); }
      throw new Error(detail || res.statusText);
    }
    const data = await res.json();
    $('upload-status').textContent = '✓ Uploaded';
    if (data.warning) showError('Warning: ' + data.warning);

    showSamplePreview(file);
    state.activeSample = data.filename;
    await fetchSamples();
    populateSampleSelect();
    updateSynthBtn();
  } catch (e) {
    $('upload-status').textContent = '';
    showError('Upload failed: ' + e.message);
  }
}

function showSamplePreview(file) {
  if (state.samplePreviewUrl) URL.revokeObjectURL(state.samplePreviewUrl);
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

  const text    = $('text-input').value.trim();
  const speed   = parseFloat($('speed-slider').value);
  const use_ssml = state.inputFormat === 'ssml';

  let body;
  if (state.mode === 'builtin') {
    const selectedVoice = state.voices.find(v => v.id === $('voice-select').value);
    const language = selectedVoice ? selectedVoice.language : 'en-us';
    body = { text, voice: $('voice-select').value, language, speed, use_ssml };
  } else {
    if (!state.activeSample) {
      showError('Please upload and verify a voice sample first.');
      showLoading(false);
      return;
    }
    const langCode = $('clone-language').value;
    const language = LANGUAGE_TAGS[langCode] || 'en-us';
    const refText  = $('ref-text').value.trim();
    const voiceId  = `clone:${state.activeSample.replace(/\.[^.]+$/, '')}`;
    body = {
      text, voice: voiceId, language, speed, use_ssml,
      reference_audio: state.activeSample,
      reference_text:  refText || null,
    };
  }

  try {
    const res = await fetch('/synthesize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      let detail;
      try { detail = (await res.json()).detail; } catch { detail = await res.text().catch(() => res.statusText); }
      throw new Error(detail || res.statusText);
    }
    renderOutput(await res.json());
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
  $('engine-label').textContent   = `engine: ${data.engine}`;
  const link = $('download-link');
  link.href     = data.audio_url;
  link.download = data.filename;
  $('output-section').classList.remove('hidden');
}

// ── Test suite ────────────────────────────────────────────────────────────────

async function runTest(testId) {
  const t    = TESTS[testId];
  const card = document.querySelector(`.test-card[data-test="${testId}"]`);
  if (!card) return;

  const outputSection = card.querySelector('.test-output');
  const errorBox      = card.querySelector('.test-error');
  const outputAudio   = card.querySelector('.test-output-audio');
  const outputMeta    = card.querySelector('.test-output-meta');

  // Reset
  outputSection.classList.add('hidden');
  errorBox.classList.add('hidden');
  errorBox.textContent = '';
  setTestStatus(card, 'running', 'Running…');

  const body = {
    text:            t.synthesisText,
    voice:           `clone:${t.referenceAudio.replace(/\.[^.]+$/, '')}`,
    language:        t.language,
    speed:           1.0,
    reference_audio: t.referenceAudio,
    reference_text:  t.referenceText,
  };

  try {
    const res = await fetch('/synthesize', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
    if (!res.ok) {
      let detail;
      try { detail = (await res.json()).detail; } catch { detail = await res.text().catch(() => res.statusText); }
      throw new Error(detail || res.statusText);
    }
    const data = await res.json();

    // Show output
    outputAudio.src = data.audio_url;
    outputAudio.load();
    outputMeta.textContent = `${data.duration_seconds.toFixed(2)}s · ${data.engine} · ${data.sample_rate} Hz`;
    outputSection.classList.remove('hidden');
    setTestStatus(card, 'pass', 'Pass ✓');

  } catch (e) {
    errorBox.textContent = '✗ ' + e.message;
    errorBox.classList.remove('hidden');
    setTestStatus(card, 'fail', 'Fail ✗');
  }
}

function setTestStatus(card, status, label) {
  const dot       = card.querySelector('.test-status-dot');
  const labelEl   = card.querySelector('.test-status-label');
  dot.dataset.status  = status;
  labelEl.textContent = label;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function showError(msg) {
  const box = $('error-box');
  box.textContent = msg;
  box.classList.remove('hidden');
}
function hideError()         { $('error-box').classList.add('hidden'); }
function showLoading(on)     { $('loading').classList.toggle('hidden', !on); }

document.addEventListener('DOMContentLoaded', init);
