# input_samples — Test Suite Reference Audio

This directory holds the pre-packaged reference audio clips used by the
**Test Results** tab in the TTS Agent UI. All three files are version-controlled
so tests are reproducible without any manual setup.

---

## Required Files

| Filename | Language | Speaker | Duration |
|---|---|---|---|
| `cloning-voice-sample-english-male-1.wav` | English (en) | Male | 5–12 s |
| `cloning-voice-sample-hindi-male-1.wav` | Hindi (hi) | Male | 5–12 s |
| `cloning-voice-sample-japanese-female-1.wav` | Japanese (ja) | Female | 5–12 s |

---

## Test Specifications

### Test 1 — English Voice Cloning

**Reference transcript** (what is spoken in `cloning-voice-sample-english-male-1.wav`):
```
Welcome to Sikkim. The land of magic, peaceful living and meditation.
Enjoy simple conversations with the friendly and helpful locals.
```

**Text to synthesise in the cloned voice**:
```
Visiting Sikkim offers a unique blend of natural beauty, cultural richness, and peaceful surroundings.
Nestled in the Himalayas, Sikkim is known for its snow-capped mountains, beautiful valleys, waterfalls,
and clean environment. Tourists can enjoy breathtaking views of Kanchenjunga, the third-highest mountain
in the world. The state is also famous for its Buddhist monasteries, colorful festivals, and warm
hospitality. Adventure lovers can experience trekking, river rafting, and mountain biking, while nature
enthusiasts can explore rich biodiversity and scenic landscapes. Sikkim's calm atmosphere and
pollution-free environment make it an ideal destination for relaxation, spiritual peace, and
unforgettable travel experiences.
```

---

### Test 2 — Hindi Voice Cloning

**Reference transcript** (what is spoken in `cloning-voice-sample-hindi-male-1.wav`):
```
नमस्कार सभी को, आज मैं समय का महत्व विषय पर कुछ शब्द कहना चाहता हूँ।
समय हमारे जीवन की सबसे कीमती चीज़ है।
```

**Text to synthesise in the cloned voice**:
```
जीवन में एक लक्ष्य होना ज़रूरी है। जब मन में एक सपना हो, तो हर सुबह उठने का कारण मिलता है।
अपने मन से पूछो — "मैं क्या बनना चाहता हूँ?" जब यह उत्तर मिल जाए, तो उसे अपनी आत्मा में बसा लो।
रास्ते में क्रोध, ईर्ष्या और निराशा आएगी। इन्हें पहचानो, इनसे लड़ो मत।
अपने भीतर की आवाज़ सुनो — यही जागरूकता तुम्हें सही राह दिखाएगी।
```

---

### Test 3 — Japanese Voice Cloning

**Reference transcript** (what is spoken in `cloning-voice-sample-japanese-female-1.wav`):
```
浜田山とは、いわゆる「超有名観光地」ではないけれど、東京のローカルで上質な
住宅街の空気感を味わえるエリアです。
```

**Text to synthesise in the cloned voice**:
```
やばい、『ホッパーズ』まじで最高だった！メイベルがビーバーのロボットに意識を移して
動物たちと一緒に生きるって設定、最初は「え、どういうこと？」ってなったけど、
見てるうちにどんどん引き込まれた。キング・ジョージがかわいすぎて、ずっと笑ってたのに、
最後は普通に泣いた。自然保護のメッセージがすごくリアルに伝わってきたのが良かった。
```

---

## Recording Guidelines

For best XTTS v2 voice cloning quality:

- **Duration**: 5–12 seconds (shorter clips are fine; longer does not improve quality)
- **Format**: 24 kHz or higher WAV (mono or stereo both work)
- **Content**: Clear, natural speech at a normal pace
- **Environment**: Minimal background noise; no music or reverb
- **Speaker**: Single speaker only
- **Consistency**: The same speaker should be used for reference and synthesis text

---

## How These Files Are Served

Files in this directory are served at:
```
GET /input-samples/{filename}
```
and are automatically searched by the XTTS v2 engine when `reference_audio` matches
a filename here (checked after `voice_samples/`).

They do **not** need to be uploaded through the UI.
