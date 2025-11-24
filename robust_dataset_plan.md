# Robust Legal Dataset Plan for Dia Architecture

## Executive Summary
**Goal:** Train a Music Foundational Model (Dia Architecture) with a "perfect optimal dataset".
**Constraints:** Strictly legal data (no scraping), specific prompt requirements (e.g., "90's synth no drums").
**Architecture Context:** The Dia model is a transformer-based architecture using the Descript Audio Codec (DAC). It expects **stereo audio (18 channels of codebooks)** at **44.1kHz**.

---

## 1. Architecture Analysis & Requirements

To build the "optimal" dataset, we must feed the model exactly what it expects.

*   **Audio Encoder:** Descript Audio Codec (DAC).
*   **Channels:** **Stereo (18 channels)**.
    *   *Note:* The model treats stereo as two 9-channel mono streams concatenated. Even mono sources should be duplicated to stereo to match the 18-channel input layer.
*   **Sample Rate:** **44.1kHz**.
*   **Context Window:**
    *   **Audio:** 600 tokens (~7 seconds at DAC's ~86Hz rate). The training loop uses a sliding window of this size.
    *   **Text:** 512 tokens.
*   **Format:** Pre-encoded `.pt` tensors are preferred for speed (`preencode_audio.py`), but raw audio (`.wav`) is supported.

---

## 2. Legal Data Sourcing Strategy

Since scraping is prohibited, we will use a **Generative-First Strategy** supplemented by **Public Domain** archives.

### A. The "Infinite" Procedural Core (For Specific Prompts)
This is the most robust way to get exactly what you asked for ("90's synth", "808 bass") without copyright issues.

1.  **DAW Automation / Headless Generation:**
    *   Use tools like **DawDreamer** (Python) or **CSound** to programmatically generate audio using VST instruments (e.g., free/open-source synths like *Surge XT* or *Vital*).
    *   **Why:** You can generate 10,000 hours of "90's synth no drums" by randomizing MIDI patterns and synth presets.
    *   **Legal Status:** You own the copyright to the output of these generations.

2.  **Specific Strategies for Your Prompts:**
    *   **"90's synth no drums":** Script a generator that loads vintage-style presets (analog saws, pads) and plays chord progressions *without* any percussion tracks.
    *   **"Hip hop beat only drums":** Use a drum machine plugin. Generate patterns at 80-100 BPM. Mute all melodic instruments.
    *   **"808 bass melody":** Use a sine-wave based synth or 808 sample. Generate low-frequency melodic lines (C1-C3 range).
    *   **"Guitar pop beat":** This is harder to synthesize. Use high-quality CC0 guitar loops or record a session guitarist playing isolated riffs, then layer with programmed drums.

### B. Public Domain & CC0 Archives
*   **Free Music Archive (FMA):** Filter strictly for **CC0** or **Public Domain**.
*   **Library of Congress:** Great for old jazz/classical (good for "foundational" understanding of harmony).
*   **Freesound.org:** Filter for CC0. Excellent for isolated instrument one-shots.

---

## 3. Dataset Structure & Pipeline

Organize your data to maximize training efficiency.

### Directory Layout
```text
/dataset
    /raw_audio
        /synth_no_drums
            track_001.wav
        /hiphop_drums
            beat_001.wav
    /audio_prompts
        track_001_prompt.txt  <-- Contains "90's synth no drums, analog pad, chord progression"
        beat_001_prompt.txt   <-- Contains "hip hop beat only drums, boom bap, 90bpm"
    /encoded_dataset          <-- Output of pre-encoding
        metadata.json
        /encoded_audio
            track_001.pt
            beat_001.pt
```

### Phase 1: Generation & Tagging
Create a generation script that produces **Audio + Prompt Pairs** simultaneously.
*   *Example:* If the script generates a drum beat at 95 BPM using a TR-808 kit, it should immediately write a text file: `hip hop beat only drums, 808 kit, 95 bpm, dry`.

### Phase 2: Pre-Encoding (Critical for Speed)
Use the existing `scripts/preencode_audio.py`. It handles the complex stereo encoding for you.

**Command:**
```bash
python scripts/preencode_audio.py \
  --audio_dir dataset/raw_audio \
  --out_dir dataset/encoded_dataset \
  --prompts_dir dataset/audio_prompts \
  --sr 44100 \
  --device cuda
```
*   **Why:** This runs the heavy DAC encoding *once* instead of every epoch, speeding up training by 10-50x.
*   **Stereo Handling:** The script automatically duplicates mono signals to stereo (lines 96-99), ensuring compatibility with the 18-channel architecture.

---

## 4. Optimization Techniques

### "Perfect Optimal" Formatting
1.  **Sliding Window Alignment:** The model trains on ~10s chunks (`finetune_acc.py` line 292).
    *   *Recommendation:* Generate/Cut audio into **30-60 second** files. This gives the model enough context to learn structure (A/B sections) but keeps file loading efficient.
2.  **Silence Trimming:** Remove silence from the start/end of generated files. Silence wastes tokens.
3.  **Normalization:** Normalize all audio to -14 LUFS or peaks at -1.0 dB. Consistent volume helps the model focus on timbre rather than gain.

### Prompt Engineering for Training
The model learns the mapping between *text* and *audio*.
*   **Bad Prompt:** "Audio file 1"
*   **Good Prompt:** "90's synth no drums"
*   **Optimal Prompt:** "90's synth no drums, analog pad, slow attack, minor chord progression, atmospheric, lo-fi"

**Strategy:** Use "Tag Soup" style prompting (comma-separated tags) as seen in the training code (`_augment_tags` function in `finetune_acc.py`).

---

## 5. Action Plan Summary

1.  **Setup Generation Rig:** Create python scripts using a library like `scipy` or `dawdreamer` to synthesize thousands of files for your specific categories (Synth, Drums, Bass).
2.  **Auto-Tag:** Ensure your generation scripts write `_prompt.txt` files automatically with the specific keywords you need ("no drums", "only drums").
3.  **Encode:** Run `preencode_audio.py` on the generated folder.
4.  **Train:** Point `finetune_acc.py` to the `--preencoded_dir`.

```bash
python dia/finetune_acc.py \
  --preencoded_dir dataset/encoded_dataset \
  --output_dir checkpoints/ \
  --batch_size 4 \
  --grad_accum_steps 4
```

