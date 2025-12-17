# Dataset Pipeline Overview

This document outlines the data loading and processing logic in `dia/dataset.py` for training the Dia stereo audio model.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐          ┌─────────────────────┐                      │
│  │   Raw Audio      │          │   Pre-encoded .pt   │                      │
│  │  (.mp3/.wav/...)│          │   (T, 18) tensors   │                      │
│  └────────┬─────────┘          └──────────┬──────────┘                      │
│           │                               │                                  │
│           ▼                               ▼                                  │
│  ┌──────────────────┐          ┌─────────────────────┐                      │
│  │  TestingDataset  │          │ PreEncodedDACDataset│                      │
│  │  (on-the-fly     │          │  (fast loading)     │                      │
│  │   DAC encoding)  │          │                     │                      │
│  └────────┬─────────┘          └──────────┬──────────┘                      │
│           │                               │                                  │
│           └───────────┬───────────────────┘                                  │
│                       ▼                                                      │
│              ┌────────────────┐                                              │
│              │   collate_fn   │ ─── Batching, padding, delay patterns,      │
│              └────────┬───────┘     attention masks, random cropping        │
│                       ▼                                                      │
│              ┌────────────────┐                                              │
│              │  Training Loop │                                              │
│              └────────────────┘                                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Two Dataset Classes

### 1. `TestingDataset` — On-the-fly DAC Encoding

**Use case:** Development, small datasets, or when pre-encoding isn't feasible.

**Warning:** Very slow due to DAC encoding blocking training. Not recommended for production.

```python
TestingDataset(
    audio_folder: Path,       # Folder containing raw audio files
    config: DiaConfig,        # Model configuration
    dac_model: dac.DAC,       # Pre-loaded DAC model instance
    use_sliding_window: bool, # Random vs fixed cropping (default: True)
    skip_tags: list,          # Filter out samples containing these text tags
    allow_empty_prompts: bool # Allow missing prompts (default: True)
)
```

**Directory Structure Expected:**
```
project/
├── audio/
│   ├── song1.mp3
│   ├── song2.wav
│   └── song3.flac
└── audio_prompts/
    ├── song1_prompt.txt
    ├── song2_prompt.txt
    └── song3_prompt.txt
```

**Process Flow:**
1. Scans `audio_folder` for files with extensions: `.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`
2. Matches each audio file with `{stem}_prompt.txt` in sibling `audio_prompts/` folder
3. On `__getitem__`:
   - Loads audio, resamples to 44.1kHz
   - Crops to target length (random or fixed based on `use_sliding_window`)
   - Encodes with DAC: each mono channel → 9 codebooks
   - Stereo: concatenate left + right → 18 codebooks
   - Mono: duplicate to fake stereo → 18 codebooks

---

### 2. `PreEncodedDACDataset` — Pre-encoded Tensors (Recommended)

**Use case:** Production training, large datasets.

**Speed:** ~100x faster than on-the-fly encoding.

```python
PreEncodedDACDataset(
    preprocessed_dir: Path,   # Directory with encoded_audio/ and metadata.json
    config: DiaConfig,        # Model configuration
    use_sliding_window: bool  # Random cropping in collate_fn (default: True)
)
```

**Directory Structure Expected:**
```
preprocessed/
├── encoded_audio/
│   ├── song1.pt          # Shape: (T, 18) int16 or long tensor
│   ├── song2.pt
│   └── song3.pt
├── prompts/              # Alternative prompt location
│   ├── song1.txt
│   ├── song2.txt
│   └── song3.txt
└── metadata.json         # {"song1.pt": {"text": "prompt...", ...}}
```

**Prompt Resolution Order:**
1. `metadata.json` → `{filename}.text` field
2. `encoded_audio/{stem}.txt` (same folder as .pt)
3. `prompts/{stem}.txt`
4. `prompts/{stem}_prompt.txt`
5. Fallback: empty string (unconditional training)

**Pre-encode with:**
```bash
python scripts/preencode_audio.py \
    --audio_dir ./audio \
    --out_dir ./preprocessed \
    --prompts_dir ./audio_prompts
```

---

## Stereo Audio Encoding

The Dia model uses **18 codebooks** for stereo audio:

```
┌─────────────────────────────────────────────────────┐
│                STEREO ENCODING                       │
├─────────────────────────────────────────────────────┤
│                                                      │
│   Raw Stereo Audio                                   │
│   ┌─────────────────────────────────────────┐       │
│   │  Left Channel   │   Right Channel       │       │
│   └────────┬────────┴────────┬──────────────┘       │
│            │                 │                       │
│            ▼                 ▼                       │
│     ┌──────────────┐   ┌──────────────┐             │
│     │ DAC Encoder  │   │ DAC Encoder  │             │
│     │   (mono)     │   │   (mono)     │             │
│     └──────┬───────┘   └──────┬───────┘             │
│            │                  │                      │
│            ▼                  ▼                      │
│     ┌──────────────┐   ┌──────────────┐             │
│     │ (T, 9) codes │   │ (T, 9) codes │             │
│     │ codebooks    │   │ codebooks    │             │
│     │   0-8        │   │   9-17       │             │
│     └──────┬───────┘   └──────┬───────┘             │
│            │                  │                      │
│            └────────┬─────────┘                      │
│                     ▼                                │
│            ┌────────────────┐                        │
│            │ torch.cat()   │                         │
│            │  → (T, 18)    │                         │
│            └────────────────┘                        │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**Mono handling:** Mono audio is duplicated to both channels, resulting in identical left/right codes.

---

## Cropping Strategy

| Mode | `use_sliding_window` | Behavior | Use Case |
|------|---------------------|----------|----------|
| **Random Crop** | `True` | Random start position each epoch | Production training — data augmentation |
| **Fixed Crop** | `False` | Always crop from start (t=0) | Debugging, overfitting tests |

**Target length:** `config.data.audio_length` (default: 600 tokens = ~7 seconds @ 44.1kHz)

**Calculation:** `samples = tokens × 512` (DAC hop_length = 512)

---

## Data Flow Through collate_fn

The `collate_fn` in `finetune_acc.py` handles:

```python
# Input: List of (text, encoded, waveform) tuples
# Output: Dictionary with batched tensors

{
    'src_tokens':        (B, 512),      # UTF-8 encoded text
    'src_positions':     (B, 512),      # Position indices
    'enc_self_attn_mask': (B, 1, 512, 512),  # Encoder attention mask
    
    'tgt_tokens':        (B, T+2, 18),  # BOS + codes + EOS + PAD
    'tgt_positions':     (B, T+2),      # Position indices
    'dec_self_attn_mask': (B, 1, T+2, T+2),  # Causal mask
    'dec_cross_attn_mask': (B, 1, T+2, 512), # Cross-attention mask
    
    'seq_lens':          [L1, L2, ...],  # Original lengths before padding
}
```

**Key operations:**
1. **Random cropping** (if `use_sliding_window=True`)
2. **Text tokenization** (UTF-8 bytes → integers)
3. **Audio padding** (to batch max length)
4. **Delay pattern application** (codebook interleaving)
5. **BOS/EOS injection** (sequence boundaries)
6. **Attention mask generation** (causal + padding aware)

---

## Configuration Reference

From `config.json`:

```json
{
    "data": {
        "text_length": 512,          // Max text tokens
        "audio_length": 600,         // Max audio tokens (~7 sec)
        "channels": 18,              // Stereo = 9 + 9 codebooks
        "text_pad_value": 0,         // Text padding token
        "audio_eos_value": 1024,     // End of sequence
        "audio_pad_value": 1025,     // Padding token
        "audio_bos_value": 1026,     // Beginning of sequence
        "delay_pattern": [0,1,2,3,4,5,6,7,8, 0,1,2,3,4,5,6,7,8]
    }
}
```

---

## Return Format

Both dataset classes return the same tuple format:

```python
(
    text_prompt: str,           # Text description/prompt
    encoded: Tensor (T, 18),    # DAC codes (stereo)
    waveform: Tensor | None     # Raw waveform (TestingDataset only)
)
```

This ensures compatibility with a single `collate_fn` regardless of which dataset is used.

---

## Best Practices

1. **Always pre-encode for training** — Use `scripts/preencode_audio.py` to avoid DAC encoding bottleneck
2. **Use sliding window for training** — Random cropping provides data augmentation
3. **Disable sliding window for debugging** — Deterministic crops help isolate issues
4. **Provide quality prompts** — Empty prompts lead to unconditional generation
5. **Use `skip_tags` to filter** — Exclude unwanted samples (e.g., "[instrumental]" for vocals training)

