# Reverting to the original KV cache logic (without a full layers rollback)

Goal: keep your current `layers.py` refactor, but restore the original KV cache behavior to unblock generation. The original cache (see `dia_original/dia/dia/state.py` and `dia_original/dia/dia/layers.py`) relies on:
- A `KVCache` that stores buffers shaped `(2*B, num_heads, max_len, head_dim)` and updates in-place via `update`/`prefill`.
- Attention that directly reads/writes that cache in-place and uses `current_idx` for decode steps.
- Inference state classes (`EncoderInferenceState`, `DecoderInferenceState`) that hold the caches and masks.

Here’s how to reintroduce that behavior without breaking the rest of your refactor.

## 1) Add the original cache and state helpers side-by-side
- Copy `dia_original/dia/dia/state.py` into `dia/state.py` (or a new name like `state_original.py` if you want to keep both). This gives you `KVCache`, `EncoderInferenceState`, `DecoderInferenceState`, `create_attn_mask`.
- If you keep both, export the chosen one in `dia/__init__.py` so imports are consistent.

## 2) Patch `layers.py` to use the original cache API for attention
- Replace the current `KVCache` class in `dia/layers.py` with the one from the original (`update`, `prefill`, `from_kv`, and 2x batch sizing if you’re using CFG).
- In `Attention.forward`, switch to the original in-place update flow:
  - For self-attn decode step: call `cache.update(Xk_BxKxSxH, Xv_BxKxSxH, current_idx)` instead of returning `new_kv_cache`.
  - For prefill: call `cache.prefill(...)`.
  - Drop `new_kv_cache` return; just return `output`.
- Reintroduce the `current_idx` argument for decode paths and wire it through `DecoderLayer` and `Decoder.decode_step` so each layer passes the index to self-attn.

## 3) Adjust masks/positions to match the original
- The original uses `create_attn_mask` and stores `enc_state.positions`, `dec_state.dec_positions`, `dec_state.casual_attn_mask`, and `dec_state.cross_attn_mask`.
- If you don’t want the full state class, at minimum align the causal/self masks and pass `current_idx` so attention can pick the right cache slot.

## 4) Minimal call-site changes
- In decode loops (e.g., `Dia.generate` or any sampling code), ensure you call `prepare_step` (or manually set `current_idx`) before each decode step, and pass `current_idx` into `decode_step` → `DecoderLayer` → `Attention`.
- If you keep CFG (2*B batches), mirror the original behavior: cache tensors sized for 2*B, and ensure positions/masks are duplicated for unconditional/conditional streams.

## 5) Keep the rest of the refactor intact
- You do NOT need to revert MLP, config schema, or dropout logic for this step. Just swap the cache class and how it is updated in attention/decoder paths.

## 6) Quick sanity checklist after patching
- Shapes: cache `k/v` should be `(batch, heads, max_len, head_dim)` (or `2*B` if CFG). No mismatched head counts.
- Decode step: `current_idx` increments each token; cache slices update correctly.
- Prefill: `prefill` fills the first `S` slots and sets `current_idx` accordingly (if you keep that field).
- Masks: causal mask applied in attention (or via `create_attn_mask`).
- A single-step generate run doesn’t error and produces logits.

If you prefer a surgical diff, reuse the original `KVCache` and attention update code from `dia_original/dia/dia/layers.py` (classes `KVCache`, `SelfAttention`/`Attention` forward) and thread `current_idx` through decoder decode paths. This gives you the old caching semantics without touching the newer activation/dropout/config changes.***
