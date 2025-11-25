# Checkpoint Evaluation Plan

## Overview

A systematic approach to evaluating and selecting the best checkpoint from a training run. This plan incorporates industry-standard practices (Google MusicLM, OpenAI Jukebox, Suno, Udio) to ensure model quality, musicality, and deployment readiness.

**Goals:**
- Identify the optimal stopping point (trade-off between convergence and overfitting).
- Quantify "musicality" and structural coherence, not just audio quality.
- Ensure diversity and lack of unintended memorization.

---

## Evaluation Dimensions

### 1. Objective Metrics (Automated)

| Metric | What it Measures | Implementation |
|--------|------------------|----------------|
| **CLAP Score** | **Text-Audio Alignment**: How well the generated audio matches the prompt. | `laion-clap` |
| **FAD (Fréchet Audio Distance)** | **Realism**: Distance between distribution of generated audio and a reference set of "real" music. Lower is better. | `frechet_audio_distance` (VGGish or CLAP embeddings) |
| **FAD (Deep Embeddings)** | **High-Level Realism**: Using modern codecs (Encodec/DAC) or CLAP embeddings for FAD often correlates better with human perception than VGGish. | Custom FAD wrapper |
| **Mel-Spectrogram Distance (LSD)**| **Signal Quality**: Log-Spectral Distance measures low-level reconstruction errors (useful for vocoder/decoder quality). | Custom (LSD) |
| **KL Divergence / Vendi Score** | **Diversity**: Measures if the model generates diverse outputs for the same prompt or across prompts (prevents mode collapse). | `vendi_score` |

### 2. Musicality & Structural Metrics (MIR)

Crucial for music generation (Suno/Udio standard). Does the model understand music theory?

| Metric | What it Measures | Implementation |
|--------|------------------|----------------|
| **Beat/Tempo Stability** | Consistency of rhythm. Does the model hold a steady tempo? | `librosa.beat` / `madmom` |
| **Key/Chord Consistency** | Tonal stability. Does the audio stay in key or modulate logically? | `chroma_cqt` correlation over time |
| **Genre Classification** | Classifiability. Can a standard classifier (e.g., PANNs) recognize the generated genre matches the prompt? | Pre-trained audio classifier |

### 3. Reference-Based Metrics (For Reconstruction/Overfit Tests)

When ground truth is available (e.g., autoencoding task or specific song overfitting):

- **Mel-spectrogram MSE**: Pixel-wise loss on spectrograms.
- **Embedding Cosine Similarity**: Distance in CLAP/Encodec latent space.
- **Waveform L1/L2**: Only if phase reconstruction is deterministic (rare in generative models).

### 4. Human Evaluation (Gold Standard)

Automated metrics often fail to capture "vibes" or long-range structure.

- **MOS (Mean Opinion Score)**: 1-5 rating on overall audio quality.
- **Musical Coherence**: 1-5 rating on "Does this sound like a structured song?" (Verse/Chorus structure).
- **Prompt Adherence**: "Did I hear the 'saxophone' requested?"
- **Artifacts**: Presence of clicking, static, or vocoder artifacts.
- **A/B Preference**: Side-by-side comparison (Checkpoint A vs B).

---

## Evaluation Protocol

### Test Set Design

We define specific "buckets" to test different capabilities:

```python
EVAL_PROMPTS = {
    # 1. Distribution Coverage (Standard styles)
    "calm_piano": "solo piano, ambient, peaceful, reverb, 80bpm",
    "electronic_dance": "edm, driving beat, synthesizer, 128bpm, club",
    
    # 2. Compositional Complexity (Harder)
    "jazz_fusion": "jazz fusion, saxophone solo, complex chords, fast tempo",
    "orchestral_build": "cinematic, orchestra, crescendo, epic, movie score",
    
    # 3. Combinatorial Generalization (OOD)
    "unseen_mix": "heavy metal mixed with bagpipes, aggressive, distortion",
    
    # 4. Temporal Consistency (Long generation test)
    "progressive_structure": "song structure, intro, verse, chorus, bridge",
}

EVAL_SEEDS = [42, 1337, 999] # Fixed seeds for apples-to-apples comparison
```

### Checkpoint Sampling Strategy

1.  **Coarse Sweep**: Every ~5k steps.
2.  **Fine Sweep**: Around loss minima (every ~1k steps).
3.  **Sanity Check**: Always check the *latest* checkpoint to catch sudden collapse.

### Generation Protocol (Inference)

1.  **Fixed Parameters**: Use `cfg_scale=3.0`, `temp=1.0` (or standardized values).
2.  **Duration**: Generate at least 10-30s to test temporal coherence (industry standard is often 10s clips for eval, but 30s+ for structure).
3.  **Naming**: `{step}_{prompt_slug}_s{seed}_cfg{cfg}.wav`

---

## Output & Reporting

Structure the results to easily compare "Step 10k" vs "Step 20k".

```
evaluation_results/
├── audio/
│   ├── step_05000/
│   ├── step_10000/
│   └── ...
├── metrics/
│   ├── summary_table.csv (cols: Step, CLAP, FAD, Diversity, Key_Stability)
│   ├── fad_scores.json
│   └── genre_confusion_matrix.png
└── reports/
    └── latest_eval.md (Auto-generated with best checkpoint recommendation)
```

---

## Advanced: Long-Form Consistency (The "Suno Test")

To ensure the model doesn't drift or lose coherence over time (e.g., tempo drifting, texture collapsing to noise):

1.  **Segmented Analysis**: Split 30s generation into three 10s chunks.
2.  **Self-Similarity**: Compute CLAP/Mel similarity between Chunk 1 and Chunk 3.
    *   *High similarity* = Consistent style.
    *   *Low similarity* = Model drifted or hallucinated.

---

## Implementation Phases

### Phase 1: The "Vibe Check" (Manual)
- [ ] Script to generate a grid of audios (Prompts x Seeds) for a given checkpoint.
- [ ] Manually listen to "Jazz" and "Piano" to catch obvious failure modes (silence, noise).

### Phase 2: Automated Basics
- [x] Implement `CLAP` scoring script.
- [x] Implement `FAD` (requires computing background stats on the training dataset).


### Phase 3: The "Musician" Check
- [ ] Add `librosa` beat tracking to check for rhythmic collapse.
- [ ] Add `vendi_score` to measure diversity (are all seeds producing the same audio?).

### Phase 4: Production Readiness
- [ ] **Memorization Check**: Heuristic check against training data (optional/advanced).
- [ ] **RTF Measurement**: Log time-to-generate vs audio duration.
- [ ] **Safety Filter**: Check for generation of silence, white noise bursts, or extremely loud artifacts (common in diffusion/flow matching).

---

## External Benchmarks (Optional)

If you want to compare your model against public research (MusicLM, AudioLDM, MusicGen):

- **MusicCaps Benchmark** (Google): The gold standard for text-to-audio.
    - *Dataset*: ~5.5k expert-annotated clips.
    - *Goal*: Report FAD and CLAP scores.
    - *Use Case*: Prove your model understands descriptive text.

- **MusicBench** (Microsoft/Research):
    - *Dataset*: ~52k clips with rich metadata (chords, beat, key).
    - *Goal*: Evaluate "control" capabilities.
    - *Use Case*: Prove your model follows musical instructions (e.g. "C Major", "120 BPM"), not just vague descriptions.

- **SongEval** (Recent):
    - *Goal*: Evaluate full-length song coherence.
    - *Use Case*: If generating >30s audio, use this to benchmark structure against human ratings.

---

## Dependencies

```txt
laion-clap
frechet-audio-distance
librosa
numpy
pandas
vendi_score  # for diversity metrics
scipy
```
