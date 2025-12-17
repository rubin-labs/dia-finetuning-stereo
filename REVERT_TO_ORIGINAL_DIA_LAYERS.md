# Reverting `dia/layers.py` and `dia/state.py` to the original Dia TTS versions

You already vendor the old code under `dia_original/dia/dia`. This guide lists what to copy and what else must change so the project compiles again with the old APIs.

## 1) Copy the original files
- Overwrite the current files with the originals:
  - `cp dia_original/dia/dia/layers.py dia/layers.py`
  - `cp dia_original/dia/dia/state.py dia/state.py`
  - Add `dia/__init__.py` exports if needed so `state` is importable.

## 2) Align the config schema
The original layers import `DecoderConfig`, `EncoderConfig`, and expect fields like `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `head_dim`, etc., under `config.encoder_config` / `config.decoder_config`.

Options:
- **Simplest:** also replace `dia/config.py` with `dia_original/dia/dia/config.py`.
- Or, manually add the old field names to the current config and re-expose `encoder_config` / `decoder_config` attributes that mirror the new `model.encoder` / `model.decoder` fields.

If you have saved JSON configs, they must use the old field names once you swap configs.

## 3) Update call sites to the old forward API
The original code relies on inference state objects from `state.py`:
- `Encoder.forward(x_ids, state: EncoderInferenceState)`
- `Decoder.forward(tgt_ids, state: DecoderInferenceState)`
- Attention modules don’t return `(output, new_kv_cache)`; caches are updated in-place via `KVCache.update` / `prefill`.

Places to sync (copy the originals from `dia_original/dia/dia` or adapt manually):
- `dia/model.py` (sampling/generation expects `KVCache`, `EncoderInferenceState`, `DecoderInferenceState`)
- Training scripts: `dia/finetune.py`, `dia/finetune_acc.py`, `dia/finetune_acc_lora.py`, `dia/finetune_olderversion.py`, `dia/convert_ckpt.py`
- Any code that builds masks/positions must construct the inference states (`create_attn_mask`, `EncoderInferenceState.new`, `DecoderInferenceState.new`).

## 4) Cache and mask behavior
- Old `KVCache` doubles batch (prefill uses `2 * batch_size`), updates in-place with `current_idx`.
- Self-attention uses `is_causal` flag and custom MPS-safe attention; cross-attention uses precomputed K/V without RoPE on queries.
- Ensure decode loops call `DecoderInferenceState.prepare_step` and pass `current_idx` into `DecoderLayer` self-attn.

## 5) Dropout/dtype/activation differences
- Original MLP is fixed SwiGLU (no dropout, no pre-norm, fixed 2-way fused projection).
- `DenseGeneral` uses the weight dtype and doesn’t cast to float32 like the current file.
- If your training scripts rely on `deterministic` flags or activation lists, those hooks disappear when reverting.

## 6) Verification checklist
After copying the originals (layers, state, config, and optionally model/training files), run a quick smoke test:
- Import and instantiate the model with an old-format config.
- Run one encoder+decoder forward with dummy data.
- If training: run a tiny batch through `finetune_acc.py` to ensure masks/caches shape-check.

## 7) Quick command summary (if fully reverting to the vendored originals)
```bash
cp dia_original/dia/dia/config.py dia/config.py
cp dia_original/dia/dia/state.py dia/state.py
cp dia_original/dia/dia/layers.py dia/layers.py
cp dia_original/dia/dia/model.py dia/model.py
```
Then adjust imports/exports if any new files were added since the fork.

Keeping this guide in the repo should make future flips between the two implementations easier. Run diffs (`diff -u dia_original/dia/dia/layers.py dia/layers.py`) to confirm the swap.
