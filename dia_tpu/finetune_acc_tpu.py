import warnings
warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated")

import argparse
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
# DDP imports removed
from torch.utils.data import DataLoader, Dataset
# autocast removed - handled by Accelerate
# clip_grad_norm_ removed - handled by Accelerate
from transformers import get_scheduler
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import parametrize
# import bitsandbytes as bnb # Removed for TPU compatibility
from tqdm import tqdm
import wandb
import time
import torchaudio
import soundfile as sf

from accelerate import Accelerator
from accelerate.utils import set_seed

# TPU-specific imports
try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

import dac

# Add path to parent for dia imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from dia.audio import build_revert_indices, revert_audio_delay


# Eval demo settings
EVAL_AUDIO_DIR = "./audio_demos"
EVAL_SAMPLE_RATE = 44100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


CODEBOOK_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

NONFINITE_HIT_LIMIT = 3
_nonfinite_hits = 0


def seed_everything(seed: int):
    """Set seeds for reproducible training."""
    set_seed(seed)
    # Additional manual seeding if needed, but set_seed covers most
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strip_weight_norms(module: torch.nn.Module) -> int:
    """
    Remove weight_norm parametrizations so XLA doesn't fall back to CPU for
    aten::_weight_norm_interface (unsupported op on TPU).
    """
    removed = 0
    for m in module.modules():
        if parametrize.is_parametrized(m) and "weight" in getattr(m, "parametrizations", {}):
            try:
                parametrize.remove_parametrizations(m, "weight", leave_parametrized=False)
                removed += 1
                continue
            except Exception:
                pass
        if hasattr(m, "weight_g") or hasattr(m, "weight_orig"):
            try:
                torch.nn.utils.remove_weight_norm(m)
                removed += 1
            except ValueError:
                pass
    return removed


# Removed DDP setup functions as they are replaced by Accelerator


@dataclass
class TrainConfig:
    epochs: int = 500
    batch_size: int = 4
    grad_accum_steps: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_steps: int = 500
    unconditional_frac: float = 0.15
    save_step: int = 100
    seed: int = 786
    runs_dir: Path = Path("runs")
    run_name: str = "audio_finetune_scratch"
    output_dir: Path = None
    no_decay_embed: bool = False


@dataclass
class DataSettings:
    text_length: int = 512
    audio_length: int = 600
    channels: int = 18
    text_pad_value: int = 0
    audio_eos_value: int = 1024
    audio_pad_value: int = 1025
    audio_bos_value: int = 1026
    delay_pattern: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8] * 2)


@dataclass
class ModelSettings:
    src_vocab_size: int = 256
    tgt_vocab_size: int = 1028
    model_dim: int = 512
    dropout: float = 0.1


class MLPBlock(nn.Module):
    """Small residual MLP block to keep the scratch model lightweight."""
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.drop(self.fc2(h))
        return x + self.drop(h)


class SimpleAudioModel(nn.Module):
    """
    Lightweight audio model trained from scratch.
    Predicts per-channel audio codes conditioned on text prompts.
    """
    def __init__(self, model_cfg: ModelSettings, data_cfg: DataSettings, num_layers: int = 4):
        super().__init__()
        self.model_dim = model_cfg.model_dim
        self.data_cfg = data_cfg
        self.text_pad_value = data_cfg.text_pad_value

        self.text_embed = nn.Embedding(
            model_cfg.src_vocab_size,
            model_cfg.model_dim,
            padding_idx=data_cfg.text_pad_value if data_cfg.text_pad_value < model_cfg.src_vocab_size else None,
        )
        self.text_pos_embed = nn.Embedding(data_cfg.text_length, model_cfg.model_dim)

        self.audio_embed = nn.Embedding(
            model_cfg.tgt_vocab_size,
            model_cfg.model_dim,
            padding_idx=data_cfg.audio_pad_value if data_cfg.audio_pad_value < model_cfg.tgt_vocab_size else None,
        )
        self.audio_pos_embed = nn.Embedding(data_cfg.audio_length + 2, model_cfg.model_dim)
        self.channel_embed = nn.Embedding(data_cfg.channels, model_cfg.model_dim)

        self.blocks = nn.ModuleList([MLPBlock(model_cfg.model_dim, model_cfg.dropout) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(model_cfg.model_dim)
        self.out_proj = nn.Linear(model_cfg.model_dim, model_cfg.tgt_vocab_size)
        self.dropout = nn.Dropout(model_cfg.dropout)

    def forward(
        self,
        src_BxS: torch.Tensor,
        tgt_BxTxC: torch.Tensor,
        src_positions: torch.Tensor | None = None,
        tgt_positions: torch.Tensor | None = None,
        enc_self_attn_mask: torch.Tensor | None = None,
        dec_self_attn_mask: torch.Tensor | None = None,
        dec_cross_attn_mask: torch.Tensor | None = None,
        enable_dropout: bool = True,
    ) -> torch.Tensor:
        B, T, C = tgt_BxTxC.shape
        device = tgt_BxTxC.device

        # Text encoding (mean pooling over non-pad tokens)
        if src_positions is None:
            src_positions = torch.arange(src_BxS.size(1), device=device).unsqueeze(0).expand_as(src_BxS)
        src_pos_emb = self.text_pos_embed(torch.clamp(src_positions, max=self.text_pos_embed.num_embeddings - 1))
        src_emb = self.text_embed(torch.clamp(src_BxS, max=self.text_embed.num_embeddings - 1)) + src_pos_emb

        if enc_self_attn_mask is not None:
            # Mask is (B, 1, S, S); derive pad mask from diagonal
            pad_mask = enc_self_attn_mask[:, 0].any(-1)  # (B, S)
        else:
            pad_mask = src_BxS.ne(self.text_pad_value)

        denom = pad_mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)
        text_ctx = (src_emb * pad_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom  # (B, 1, D)

        # Audio token embeddings
        if tgt_positions is None:
            tgt_positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        tgt_pos_emb = self.audio_pos_embed(torch.clamp(tgt_positions, max=self.audio_pos_embed.num_embeddings - 1))
        tgt_pos_emb = tgt_pos_emb.unsqueeze(2)  # (B, T, 1, D)

        channel_ids = torch.arange(C, device=device)
        channel_emb = self.channel_embed(channel_ids).view(1, 1, C, -1)

        audio_emb = self.audio_embed(torch.clamp(tgt_BxTxC, max=self.audio_embed.num_embeddings - 1))
        x = audio_emb + tgt_pos_emb + channel_emb + text_ctx.unsqueeze(2)
        x = self.dropout(x) if enable_dropout else x

        for block in self.blocks:
            x = block(x)

        x = self.out_norm(x)
        logits = self.out_proj(x)
        return logits


class AudioPromptDataset(Dataset):
    """
    Loads raw audio + prompt pairs and encodes audio with DAC on the fly.
    """
    def __init__(
        self,
        audio_folder: Path,
        config: DataSettings,
        dac_model: dac.DAC,
        use_sliding_window: bool = True,
        ignore_missing_prompts: bool = True,
        allow_empty_prompts: bool = True,
    ):
        self.audio_folder = Path(audio_folder)
        self.prompts_folder = self.audio_folder.parent / "audio_prompts"
        self.config = config
        self.dac_model = dac_model
        self.use_sliding_window = use_sliding_window
        self.allow_empty_prompts = allow_empty_prompts

        audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg']
        self.audio_files = []
        for ext in audio_extensions:
            self.audio_files.extend(list(self.audio_folder.glob(f"*{ext}")))

        self.valid_files = []
        skipped_missing = 0
        for audio_file in self.audio_files:
            prompt_file = self.prompts_folder / f"{audio_file.stem}_prompt.txt"
            if not prompt_file.exists():
                if allow_empty_prompts:
                    self.valid_files.append(audio_file)
                    continue
                if ignore_missing_prompts:
                    skipped_missing += 1
                    continue
                raise FileNotFoundError(f"Missing prompt file: {prompt_file}")
            self.valid_files.append(audio_file)

        if not self.valid_files:
            raise ValueError(f"No valid audio files found in {self.audio_folder}. Skipped {skipped_missing} missing prompts.")

    def __len__(self) -> int:
        return len(self.valid_files)

    def _encode_mono_channel(self, wav_1xS: torch.Tensor) -> torch.Tensor:
        device = next(self.dac_model.parameters()).device
        audio_tensor = self.dac_model.preprocess(wav_1xS.unsqueeze(0), 44100).to(device)
        _, enc, *_ = self.dac_model.encode(audio_tensor, n_quantizers=None)
        enc = enc.squeeze(0).transpose(0, 1).to(torch.long)  # (T, 9)
        return enc

    def __getitem__(self, idx: int):
        audio_file = self.valid_files[idx]
        prompt_file = self.prompts_folder / f"{audio_file.stem}_prompt.txt"

        if prompt_file.exists():
            with open(prompt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            text = ""

        waveform, sr = torchaudio.load(audio_file)
        if sr != 44100:
            waveform = torchaudio.functional.resample(waveform, sr, 44100)

        target_length_samples = int(self.config.audio_length * 512)
        total_samples = waveform.shape[1]
        if total_samples > target_length_samples:
            if self.use_sliding_window:
                start_sample = random.randint(0, total_samples - target_length_samples)
            else:
                start_sample = 0
            waveform = waveform[:, start_sample:start_sample + target_length_samples]

        with torch.no_grad():
            num_channels = waveform.shape[0]
            if num_channels > 2:
                raise ValueError(f"Audio file {audio_file} has {num_channels} channels. Only mono/stereo supported.")
            elif num_channels == 2:
                enc_L = self._encode_mono_channel(waveform[0:1, :])
                enc_R = self._encode_mono_channel(waveform[1:2, :])
            else:
                mono = waveform[0:1, :]
                enc_L = self._encode_mono_channel(mono)
                enc_R = enc_L.clone()

            encoded = torch.cat([enc_L, enc_R], dim=1)  # (T, 18) for stereo

        encoded = encoded.to(torch.long).cpu()

        if encoded.size(1) < self.config.channels:
            pad_cols = self.config.channels - encoded.size(1)
            pad_tensor = torch.full((encoded.size(0), pad_cols), self.config.audio_pad_value, dtype=encoded.dtype, device=encoded.device)
            encoded = torch.cat([encoded, pad_tensor], dim=1)
        elif encoded.size(1) > self.config.channels:
            encoded = encoded[:, : self.config.channels]

        waveform = waveform.unsqueeze(0)
        return text, encoded, waveform


class PreEncodedDACDataset(Dataset):
    """
    Dataset for pre-encoded DAC files (.pt) with optional prompts.
    """
    def __init__(self, preprocessed_dir: Path, config: DataSettings, use_sliding_window: bool = True):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.config = config
        self.use_sliding_window = use_sliding_window

        metadata_file = self.preprocessed_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        encoded_dir = self.preprocessed_dir / "encoded_audio"
        if encoded_dir.exists():
            self.encoded_files = list(encoded_dir.glob("*.pt"))
        else:
            self.encoded_files = list(self.preprocessed_dir.glob("*.pt"))
            encoded_dir = self.preprocessed_dir

        if not self.encoded_files:
            raise FileNotFoundError(f"No .pt files found in {self.preprocessed_dir} or {encoded_dir}")

    def __len__(self) -> int:
        return len(self.encoded_files)

    def __getitem__(self, idx: int):
        encoded_file = self.encoded_files[idx]
        encoded = torch.load(encoded_file, map_location='cpu')

        target_length = self.config.audio_length
        if encoded.shape[0] > target_length:
            if self.use_sliding_window:
                start = random.randint(0, encoded.shape[0] - target_length)
                encoded = encoded[start : start + target_length]
            else:
                encoded = encoded[:target_length]

        if encoded.size(1) < self.config.channels:
            pad_cols = self.config.channels - encoded.size(1)
            pad_tensor = torch.full((encoded.size(0), pad_cols), self.config.audio_pad_value, dtype=encoded.dtype, device=encoded.device)
            encoded = torch.cat([encoded, pad_tensor], dim=1)
        elif encoded.size(1) > self.config.channels:
            encoded = encoded[:, : self.config.channels]

        if encoded_file.name in self.metadata and 'text' in self.metadata[encoded_file.name]:
            text_prompt = self.metadata[encoded_file.name]['text']
        else:
            prompt_path = encoded_file.with_suffix('.txt')
            prompt_path_alt = self.preprocessed_dir / "prompts" / f"{encoded_file.stem}.txt"
            prompt_path_alt2 = self.preprocessed_dir / "prompts" / f"{encoded_file.stem}_prompt.txt"

            if prompt_path.exists():
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    text_prompt = f.read().strip()
            elif prompt_path_alt.exists():
                with open(prompt_path_alt, 'r', encoding='utf-8') as f:
                    text_prompt = f.read().strip()
            elif prompt_path_alt2.exists():
                with open(prompt_path_alt2, 'r', encoding='utf-8') as f:
                    text_prompt = f.read().strip()
            else:
                text_prompt = ""

        return text_prompt, encoded.to(torch.long), None


def build_delay_indices(B: int, T: int, C: int, delay_pattern: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute indices for shifting audio tokens by channel-specific delays.
    Returns (t_idx_BxTxC, indices_BTCx3).
    """
    delay_arr = torch.tensor(delay_pattern, dtype=torch.int32)

    t_idx_BxT = torch.broadcast_to(
        torch.arange(T, dtype=torch.int32)[None, :],
        [B, T],
    )
    t_idx_BxTxC = t_idx_BxT[..., None] - delay_arr.view(1, 1, C)

    b_idx_BxTxC = torch.broadcast_to(
        torch.arange(B, dtype=torch.int32).view(B, 1, 1),
        [B, T, C],
    )
    c_idx_BxTxC = torch.broadcast_to(
        torch.arange(C, dtype=torch.int32).view(1, 1, C),
        [B, T, C],
    )

    t_clamped_BxTxC = torch.clamp(t_idx_BxTxC, 0, T - 1)

    indices_BTCx3 = torch.stack(
        [
            b_idx_BxTxC.reshape(-1),
            t_clamped_BxTxC.reshape(-1),
            c_idx_BxTxC.reshape(-1),
        ],
        dim=1,
    ).long()

    return t_idx_BxTxC, indices_BTCx3


def apply_audio_delay(
    audio_BxTxC: torch.Tensor,
    pad_value: int,
    bos_value: int,
    precomp: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    Apply delay pattern to audio tokens using precomputed indices.
    """
    device = audio_BxTxC.device
    t_idx_BxTxC, indices_BTCx3 = precomp
    t_idx_BxTxC = t_idx_BxTxC.to(device)
    indices_BTCx3 = indices_BTCx3.to(device)

    gathered_flat = audio_BxTxC[indices_BTCx3[:, 0], indices_BTCx3[:, 1], indices_BTCx3[:, 2]]
    gathered_BxTxC = gathered_flat.view(audio_BxTxC.shape)

    mask_bos = t_idx_BxTxC < 0
    mask_pad = t_idx_BxTxC >= audio_BxTxC.shape[1]

    bos_tensor = torch.tensor(bos_value, dtype=audio_BxTxC.dtype, device=device)
    pad_tensor = torch.tensor(pad_value, dtype=audio_BxTxC.dtype, device=device)

    return torch.where(mask_bos, bos_tensor, torch.where(mask_pad, pad_tensor, gathered_BxTxC))


def decode_dac(dac_model, codebook: torch.Tensor) -> torch.Tensor:
    """Decode DAC codes to audio waveform."""
    device = next(dac_model.parameters()).device
    codebook = codebook.to(device)
    z, _, _ = dac_model.quantizer.from_codes(codebook)
    audio = dac_model.decode(z)
    return audio.squeeze(0).squeeze(0)  # Remove batch and channel dims


def codebook_to_audio_simple(generated_codes: torch.Tensor, dac_model, delay_pattern: list[int], C: int = 18):
    """Process codebooks to generate audio. Supports mono (9) and stereo (18)."""
    # Remove BOS token if present
    if generated_codes.dim() == 2:
        # Shape: (T, C) -> add batch dim
        generated_codes = generated_codes.unsqueeze(0)
    
    # generated_codes shape: (B, T, C)
    B, T_len, num_channels = generated_codes.shape
    
    # Remove first token (BOS)
    generated_codes = generated_codes[:, 1:]
    seq_length = generated_codes.shape[1]
    
    if seq_length < 2:
        return None
    
    # Build revert indices for the delay pattern
    t_idx_BxTxC, indices_BTCx3 = build_revert_indices(B=B, T=seq_length, C=num_channels, delay_pattern=delay_pattern)
    
    # Revert the delay pattern
    reverted_codebook = revert_audio_delay(
        audio_BxTxC=generated_codes,
        pad_value=0,
        precomp=(t_idx_BxTxC, indices_BTCx3),
        T=seq_length,
    )
    
    # Trim end (delay artifacts)
    trim_amount = min(30, reverted_codebook.shape[1] - 1)
    if trim_amount > 0:
        reverted_codebook = reverted_codebook[:, :-trim_amount, :]
    
    # Shape: [B, T, C] -> [B, C, T] for DAC
    codebook = reverted_codebook.permute(0, 2, 1)
    
    # Clamp to valid range
    codebook = torch.clamp(codebook, 0, 1023)
    
    total_codebooks = codebook.shape[1]
    if total_codebooks == 9:
        return decode_dac(dac_model, codebook)
    elif total_codebooks == 18:
        left_codes = codebook[:, :9, :]
        right_codes = codebook[:, 9:, :]
        left_audio = decode_dac(dac_model, left_codes)
        right_audio = decode_dac(dac_model, right_codes)
        return torch.stack([left_audio, right_audio], dim=0)
    else:
        raise ValueError(f"Unsupported codebook channels {total_codebooks}; expected 9 or 18.")


def generate_audio_simple(
    model: SimpleAudioModel,
    data_cfg: DataSettings,
    device: torch.device,
    text: str = "",
    temperature: float = 0.0,
    max_steps: int | None = None,
) -> torch.Tensor:
    """
    Simple autoregressive generation for SimpleAudioModel.
    Returns generated codes of shape (T, C).
    Uses torch.no_grad() instead of inference_mode() for TPU compatibility.
    """
    logger.info(f"[DEBUG generate_audio_simple] Starting generation, max_steps={max_steps}")
    model.eval()
    
    max_len = max_steps if max_steps else data_cfg.audio_length
    C = data_cfg.channels
    bos_val = data_cfg.audio_bos_value
    eos_val = data_cfg.audio_eos_value
    pad_val = data_cfg.audio_pad_value
    
    logger.info(f"[DEBUG generate_audio_simple] max_len={max_len}, C={C}, device={device}")
    
    with torch.no_grad():
        # Encode text prompt
        logger.info(f"[DEBUG generate_audio_simple] Encoding text prompt...")
        max_text = data_cfg.text_length
        pad_tok = data_cfg.text_pad_value
        b_full = text.encode('utf-8')
        bts = b_full[:max_text]
        arr = list(bts) + [pad_tok] * (max_text - len(bts))
        src = torch.tensor(arr, dtype=torch.long, device=device).unsqueeze(0)  # (1, S)
        src_pos = torch.arange(max_text, device=device).unsqueeze(0)
        src_pad = src.ne(pad_tok)
        enc_self_attn_mask = (src_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)
        
        # Start with BOS token for all channels
        generated = torch.full((1, 1, C), bos_val, dtype=torch.long, device=device)
        logger.info(f"[DEBUG generate_audio_simple] Starting autoregressive loop, max_len={max_len}")
        
        for step in range(max_len):
            if step % 50 == 0:
                logger.info(f"[DEBUG generate_audio_simple] Step {step}/{max_len}, generated shape={generated.shape}")
            
            T_cur = generated.shape[1]
            tgt_pos = torch.arange(T_cur, device=device).unsqueeze(0)
            tgt_pad = generated.ne(pad_val).any(-1)
            causal = torch.tril(torch.ones((T_cur, T_cur), dtype=torch.bool, device=device))
            dec_self_attn_mask = (tgt_pad.unsqueeze(2) & tgt_pad.unsqueeze(1) & causal).unsqueeze(1)
            dec_cross_attn_mask = (tgt_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)
            
            logger.debug(f"[DEBUG generate_audio_simple] Step {step}: calling model forward...")
            logits = model(
                src_BxS=src,
                tgt_BxTxC=generated,
                src_positions=src_pos,
                tgt_positions=tgt_pos,
                enc_self_attn_mask=enc_self_attn_mask,
                dec_self_attn_mask=dec_self_attn_mask,
                dec_cross_attn_mask=dec_cross_attn_mask,
                enable_dropout=False,
            )  # (B, T, C, V)
            logger.debug(f"[DEBUG generate_audio_simple] Step {step}: model forward returned, logits shape={logits.shape}")
            
            # Get logits for last position
            last_logits = logits[:, -1, :, :]  # (B, C, V)
            
            # Sample next tokens
            if temperature == 0.0:
                next_tokens = torch.argmax(last_logits, dim=-1)  # (B, C)
            else:
                probs = F.softmax(last_logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, C)
            
            # Check for EOS on first codebook
            if (next_tokens[0, 0] == eos_val).item():
                logger.info(f"[DEBUG generate_audio_simple] EOS detected at step {step}, stopping")
                break
            
            # Append to generated
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
        
        logger.info(f"[DEBUG generate_audio_simple] Generation complete, final shape={generated.shape}")
    
    return generated.squeeze(0)  # (T, C)


def eval_demo_step(
    model,
    data_cfg: DataSettings,
    dac_model,
    global_step: int,
    device: torch.device,
    accelerator: Accelerator,
    train_cfg: TrainConfig,
):
    """
    Generate 3 eval demos with temp=0, cfg=0, prompt="" for overfitting testing.
    """
    if not accelerator.is_main_process:
        return
    
    logger.info(f"Starting eval demo generation at step {global_step}")
    
    # Create audio demo directory
    audio_dir = Path(EVAL_AUDIO_DIR)
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Unwrap model if needed
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()
    
    audio_samples = {}
    seeds = [train_cfg.seed, train_cfg.seed + 1, train_cfg.seed + 2]
    
    try:
        for i, s in enumerate(seeds):
            try:
                logger.info(f"[DEBUG] Starting demo {i+1}/3, seed={s}")
                seed_everything(s)
                logger.info(f"[DEBUG] Seed set, calling generate_audio_simple...")
                logger.info(f"Generating demo {i+1}/3 (seed={s}, temp=0, cfg=0, prompt='')")
                
                # Generate audio codes
                logger.info(f"[DEBUG] About to call generate_audio_simple with max_steps={data_cfg.audio_length}")
                generated_codes = generate_audio_simple(
                    model=unwrapped_model,
                    data_cfg=data_cfg,
                    device=device,
                    text="",
                    temperature=0.0,
                    max_steps=data_cfg.audio_length,
                )
                logger.info(f"[DEBUG] generate_audio_simple returned, shape={generated_codes.shape if generated_codes is not None else None}")
                
                if generated_codes is None or generated_codes.shape[0] < 2:
                    logger.warning(f"Demo {i+1}: generation too short, skipping")
                    continue
                
                # Decode to audio
                logger.info(f"[DEBUG] About to decode with codebook_to_audio_simple...")
                audio = codebook_to_audio_simple(
                    generated_codes,
                    dac_model,
                    data_cfg.delay_pattern,
                    C=data_cfg.channels,
                )
                logger.info(f"[DEBUG] codebook_to_audio_simple returned, audio shape={audio.shape if audio is not None else None}")
                
                if audio is None:
                    logger.warning(f"Demo {i+1}: decoding failed, skipping")
                    continue
                
                # Convert to numpy
                logger.info(f"[DEBUG] Converting audio to numpy...")
                arr = audio.detach().cpu().numpy()
                if arr.ndim == 2 and arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
                    arr = arr.T  # (C, T) -> (T, C) for soundfile
                
                # Save audio file
                logger.info(f"[DEBUG] Saving audio file...")
                audio_filename = f"step_{global_step}_demo{i+1}_seed{s}.wav"
                audio_path = audio_dir / audio_filename
                sf.write(audio_path, arr, EVAL_SAMPLE_RATE)
                logger.info(f"Saved demo audio: {audio_path}")
                
                # Add to wandb
                logger.info(f"[DEBUG] Adding to wandb...")
                audio_samples[f"eval_audio/demo{i+1}_seed{s}"] = wandb.Audio(
                    arr, sample_rate=EVAL_SAMPLE_RATE,
                    caption=f"temp=0, seed={s}"
                )
                logger.info(f"[DEBUG] Demo {i+1}/3 completed successfully")
                
            except Exception as e:
                logger.exception(f"Error generating demo {i+1} (seed={s}): {e}")
                continue
        
        # Log all audio samples to wandb
        if audio_samples:
            logger.info(f"Logging {len(audio_samples)} audio samples to wandb")
            wandb.log(audio_samples, step=global_step)
        else:
            logger.warning("No audio samples generated for logging")
            
    except Exception as e:
        logger.exception(f"Eval demo generation failed: {e}")
    finally:
        unwrapped_model.train()
        logger.info(f"Completed eval demo generation at step {global_step}")


def get_args() -> argparse.Namespace:
    train_defaults = TrainConfig()
    data_defaults = DataSettings()
    model_defaults = ModelSettings()

    parser = argparse.ArgumentParser(
        description="Train an audio model from scratch with Accelerate/TPU (arg-only, no config files)."
    )
    parser.add_argument("--audio_folder", type=Path,
                        help="Path to audio folder (expects audio_prompts folder at same level).")
    parser.add_argument("--preencoded_dir", type=Path,
                        help="Directory with pre-encoded DAC codes (encoded_audio/*.pt) and optional metadata.json.")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Output directory for checkpoints.")
    parser.add_argument("--run_name", type=str, default=train_defaults.run_name,
                        help="Run name for logging/checkpoints.")
    parser.add_argument("--seed", type=int, default=train_defaults.seed,
                        help="Random seed for reproducibility.")
    parser.add_argument("--half", action="store_true", help="enable bf16 mixed precision (TPU-safe)")
    parser.add_argument("--compile", action="store_true", help="torch compile model")
    parser.add_argument("--wandb_project", type=str, default="audio-finetuning",
                        help="Weights & Biases project name.")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity/team name.")
    parser.add_argument("--epochs", type=int, default=train_defaults.epochs,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=train_defaults.batch_size,
                        help="Batch size per device.")
    parser.add_argument("--grad_accum_steps", type=int, default=train_defaults.grad_accum_steps,
                        help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=train_defaults.learning_rate,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=train_defaults.weight_decay,
                        help="AdamW weight decay coefficient.")
    parser.add_argument("--warmup_steps", type=int, default=train_defaults.warmup_steps,
                        help="Number of warmup steps.")
    parser.add_argument("--unconditional_frac", type=float, default=train_defaults.unconditional_frac,
                        help="Fraction of unconditional training steps.")
    parser.add_argument("--no_decay_embed", action="store_true", default=train_defaults.no_decay_embed,
                        help="Exclude nn.Embedding parameters from weight decay")
    parser.add_argument("--src_vocab_size", type=int, default=model_defaults.src_vocab_size,
                        help="Source vocab size for text prompts.")
    parser.add_argument("--tgt_vocab_size", type=int, default=model_defaults.tgt_vocab_size,
                        help="Target vocab size for audio codes.")
    parser.add_argument("--model_dim", type=int, default=model_defaults.model_dim,
                        help="Hidden size for the lightweight audio model.")
    parser.add_argument("--dropout", type=float, default=model_defaults.dropout,
                        help="Dropout used inside the lightweight audio model.")
    parser.add_argument("--text_length", type=int, default=data_defaults.text_length,
                        help="Max text tokens (bytes).")
    parser.add_argument("--audio_length", type=int, default=data_defaults.audio_length,
                        help="Audio token length window.")
    parser.add_argument("--no_sliding_window", action="store_true",
                        help="Disable random sliding window; always crop from the start of each clip.")
    parser.add_argument("--channels", type=int, default=data_defaults.channels,
                        help="Number of audio code channels.")
    parser.add_argument("--text_pad_value", type=int, default=data_defaults.text_pad_value,
                        help="Pad token for text.")
    parser.add_argument("--audio_eos_value", type=int, default=data_defaults.audio_eos_value,
                        help="EOS token for audio.")
    parser.add_argument("--audio_pad_value", type=int, default=data_defaults.audio_pad_value,
                        help="Pad token for audio.")
    parser.add_argument("--audio_bos_value", type=int, default=data_defaults.audio_bos_value,
                        help="BOS token for audio.")
    parser.add_argument(
        "--delay_pattern",
        type=str,
        default=",".join(str(v) for v in data_defaults.delay_pattern),
        help="Comma-separated per-channel delays (length should match --channels).",
    )
    parser.add_argument(
        "--eval_every_step",
        type=int,
        default=0,
        help="Generate eval demos every N steps. 0 = disabled.",
    )
    
    args = parser.parse_args()

    if not args.audio_folder and not args.preencoded_dir:
        parser.error("Specify either --audio_folder or --preencoded_dir.")
    if args.audio_folder and args.preencoded_dir:
        parser.error("Use only one of --audio_folder or --preencoded_dir.")
    
    return args


def _parse_delay_pattern(raw: str | None, fallback: list[int]) -> list[int]:
    if raw is None:
        return list(fallback)
    try:
        pattern = [int(v.strip()) for v in raw.split(",") if v.strip() != ""]
    except ValueError as exc:
        raise ValueError(f"Invalid --delay_pattern '{raw}'. Use comma-separated integers.") from exc
    return pattern if pattern else list(fallback)


def collate_fn(batch, config: DataSettings, device: torch.device, use_sliding_window: bool = True):
    texts, encodings, waveforms = zip(*batch)

    window_size = config.audio_length
    cropped_encodings = []
    for e in encodings:
        if e.size(0) > window_size:
            if use_sliding_window:
                start = random.randint(0, e.size(0) - window_size)
            else:
                start = 0
            cropped_encodings.append(e[start : start + window_size])
        else:
            cropped_encodings.append(e)
    encodings = cropped_encodings

    max_text = config.text_length
    pad_tok = config.text_pad_value
    text_ids = []
    for txt in texts:
        b_full = txt.encode('utf-8')
        bts = b_full[:max_text]
        arr = list(bts) + [pad_tok] * (max_text - len(bts))
        text_ids.append(torch.tensor(arr, dtype=torch.long))
    src = torch.stack(text_ids).to(device)
    src_pos = torch.arange(max_text, device=device).unsqueeze(0).expand(src.size(0), -1)
    src_pad = src.ne(pad_tok)
    enc_self_attn_mask = (src_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)

    # For TPU efficiency, we must have fixed tensor shapes to prevent XLA recompilation.
    # We force the batch length to always match the window_size (config.data.audio_length).
    batch_max = window_size
    
    padded_encodings = []
    for e in encodings:
        if e.size(0) < batch_max:
            pad_length = batch_max - e.size(0)
            pad_value = config.audio_pad_value
            padding = torch.full((pad_length, e.size(1)), pad_value, dtype=e.dtype, device=e.device)
            padded_e = torch.cat([e, padding], dim=0)
        else:
            padded_e = e
        padded_encodings.append(padded_e)
    
    seq_lens = [e.size(0) for e in encodings]
    codes = torch.stack(padded_encodings).to(device)

    B, T, C = codes.shape
    t_idx, idxs = build_delay_indices(B, T, C, config.delay_pattern)
    delayed = apply_audio_delay(
        codes,
        config.audio_pad_value,
        config.audio_bos_value,
        (t_idx, idxs)
    )

    max_tgt_len = batch_max + 2
    pad_val = config.audio_pad_value
    bos_val = config.audio_bos_value
    eos_val = config.audio_eos_value

    tgt = torch.full((B, max_tgt_len, C), pad_val, dtype=torch.long, device=device)
    tgt[:, 0, :] = bos_val
    tgt_lens = []
    for i, L in enumerate(seq_lens):
        tgt[i, 1:1 + L, :] = delayed[i, :L, :]
        tgt[i, 1 + L, :] = eos_val
        tgt_lens.append(1 + L + 1)

    tgt_pos = torch.arange(max_tgt_len, device=device).unsqueeze(0).expand(B, -1)
    tgt_pad = tgt.ne(pad_val).any(-1)

    causal = torch.tril(torch.ones((max_tgt_len, max_tgt_len),
                                    dtype=torch.bool,
                                    device=device))
    dec_self_attn_mask = (tgt_pad.unsqueeze(2) & tgt_pad.unsqueeze(1) & causal).unsqueeze(1)
    dec_cross_attn_mask = (tgt_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)

    return {
        'src_tokens': src,
        'src_positions': src_pos,
        'enc_self_attn_mask': enc_self_attn_mask,
        'tgt_tokens': tgt,
        'tgt_positions': tgt_pos,
        'dec_self_attn_mask': dec_self_attn_mask,
        'dec_cross_attn_mask': dec_cross_attn_mask,
        'waveforms': waveforms,
        'raw_text': texts[0],
    'tgt_lens': torch.tensor(tgt_lens, dtype=torch.long, device=device),
    }

from functools import partial

def setup_loaders(dataset, data_cfg: DataSettings, train_cfg: TrainConfig, use_sliding_window: bool = True):
    collate = partial(collate_fn, config=data_cfg, device=torch.device("cpu"), use_sliding_window=use_sliding_window)
    
    ds_len = len(dataset)
    if ds_len == 0:
        raise ValueError("Dataset is empty. Check your --audio_folder/--preencoded_dir paths and filtering settings.")

    train_ds = dataset

    # Accelerator handles distribution, we just provide standard loaders
    # If the dataset is smaller than the requested batch size, drop_last=True would yield 0 steps.
    drop_last = True
    if len(train_ds) < train_cfg.batch_size:
        drop_last = False
        logger.warning(
            "Dataset size (%d) is smaller than batch_size (%d); disabling drop_last so training still runs.",
            len(train_ds),
            train_cfg.batch_size,
        )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=train_cfg.batch_size, 
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        pin_memory=False,
        drop_last=drop_last,
        persistent_workers=False,
    )
    
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError(
            f"DataLoader produced zero steps (len(train_ds)={len(train_ds)}, "
            f"batch_size={train_cfg.batch_size}, drop_last={drop_last}). "
            "Reduce batch_size or disable drop_last."
        )
    train_loader.steps_per_epoch = steps_per_epoch
    
    return train_loader



def setup_optimizer(model, train_cfg):
    norm_types = [
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
    ]
    rmsnorm_cls = getattr(torch.nn.modules.normalization, "RMSNorm", None)
    if rmsnorm_cls is not None:
        norm_types.append(rmsnorm_cls)

    no_decay_params, decay_params = [], []
    seen = set()
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            is_bias = name.endswith("bias")
            is_norm = any(isinstance(module, nt) for nt in norm_types)
            is_embed = isinstance(module, torch.nn.Embedding)
            lname = name.lower()
            is_lora_alpha = ("lora" in lname and "alpha" in lname) or lname.endswith("lora_alpha")
            if is_bias or is_norm or is_lora_alpha or (train_cfg.no_decay_embed and is_embed):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
    for _, p in model.named_parameters():
        if p.requires_grad and id(p) not in seen:
            decay_params.append(p)
            seen.add(id(p))

    param_groups = [
        {"params": decay_params, "weight_decay": train_cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Use standard AdamW for TPU compatibility instead of bnb 8-bit
    opt = optim.AdamW(
        param_groups,
        lr=train_cfg.learning_rate,
        weight_decay=0.0, # Weight decay handled in param_groups
        betas=(0.9, 0.999)
    )
    return opt


def setup_scheduler(opt, train_cfg, steps_per_epoch: int):
    total_training_steps = steps_per_epoch * train_cfg.epochs
    return get_scheduler(
        'cosine', opt,
        num_warmup_steps=train_cfg.warmup_steps // train_cfg.grad_accum_steps,
        num_training_steps=total_training_steps // train_cfg.grad_accum_steps
    )




def train(model, data_cfg: DataSettings, dataset, train_cfg: TrainConfig, args, accelerator: Accelerator, dac_model=None):
    
    if accelerator.is_main_process:
        train_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        (train_cfg.runs_dir / train_cfg.run_name).mkdir(parents=True, exist_ok=True)
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=train_cfg.run_name,
            config={
                "model": "scratch-audio",
                "dataset_size": len(dataset) if hasattr(dataset, '__len__') else "streaming",
                "epochs": train_cfg.epochs,
                "batch_size": train_cfg.batch_size,
                "grad_accum_steps": train_cfg.grad_accum_steps,
                "learning_rate": train_cfg.learning_rate,
                "warmup_steps": train_cfg.warmup_steps,
                "unconditional_frac": train_cfg.unconditional_frac,
                "seed": train_cfg.seed,
                "num_processes": accelerator.num_processes,
                "mixed_precision": accelerator.mixed_precision,
                "model_dim": getattr(model, "model_dim", None),
                "src_vocab_size": getattr(args, "src_vocab_size", None),
                "tgt_vocab_size": getattr(args, "tgt_vocab_size", None),
            }
        )
    
    # No need for barrier or manual DDP wrap or manual device move
    
    # Note: setup_loaders now returns a standard DataLoader; Accelerator wraps it later
    use_sliding_window = not getattr(args, "no_sliding_window", False)
    train_loader = setup_loaders(dataset, data_cfg, train_cfg, use_sliding_window)
    opt = setup_optimizer(model, train_cfg)

    # Cache lengths before Accelerator potentially shards loaders
    pre_steps_per_epoch = getattr(train_loader, 'steps_per_epoch', None)
    if pre_steps_per_epoch is None:
        try:
            pre_steps_per_epoch = len(train_loader)
        except Exception:
            pre_steps_per_epoch = None
    pre_dataset_len = None
    try:
        pre_dataset_len = len(train_loader.dataset)
    except Exception:
        pass

    # Decide whether to shard the dataloaders; for tiny datasets (e.g., 1 sample) we disable sharding
    shard_dataloaders = True
    if pre_dataset_len == 1:
        shard_dataloaders = False
        if accelerator.is_main_process:
            logger.warning("Dataset has exactly 1 sample; disabling sharding so every process reads it.")
    if shard_dataloaders and pre_dataset_len is not None and pre_dataset_len < accelerator.num_processes:
        shard_dataloaders = False
        if accelerator.is_main_process:
            logger.warning(
                "Dataset has %d samples but %d processes; disabling data sharding so each device sees all data.",
                pre_dataset_len,
                accelerator.num_processes,
            )

    # PREPARE EVERYTHING WITH ACCELERATOR
    if shard_dataloaders:
        model, opt, train_loader = accelerator.prepare(
            model, opt, train_loader
        )
        use_xla_native = False
    else:
        # No sharding: keep original loaders; run native XLA stepping
        model = model.to(accelerator.device)
        use_xla_native = True
        if pre_steps_per_epoch is not None:
            train_loader.steps_per_epoch = pre_steps_per_epoch

    # Note: Don't wrap with MpDeviceLoader when using Accelerate - they conflict
    # Accelerate should handle TPU data loading internally

    model.train()
    
    # CRITICAL FOR TPU: Initialize optimizer state before training to avoid
    # graph recompilation on step 2. AdamW lazily creates momentum/variance
    # tensors on first opt.step(), which changes the graph structure.
    if HAS_XLA:
        for param_group in opt.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    state = opt.state[p]
                    if len(state) == 0:
                        device = p.device
                        state['step'] = torch.zeros((), device=device)
                        state['exp_avg'] = torch.zeros_like(p, device=device)
                        state['exp_avg_sq'] = torch.zeros_like(p, device=device)

    if shard_dataloaders:
        steps_per_epoch = getattr(train_loader, 'steps_per_epoch', None)
        if steps_per_epoch is None:
            try:
                steps_per_epoch = len(train_loader)
            except Exception:
                steps_per_epoch = None
        dataset_len = None
        try:
            dataset_len = len(train_loader.dataset)
        except Exception:
            dataset_len = pre_dataset_len
    else:
        steps_per_epoch = pre_steps_per_epoch
        dataset_len = pre_dataset_len

    if steps_per_epoch is None or steps_per_epoch == 0:
        raise RuntimeError(
            f"Training dataloader is empty (steps_per_epoch={steps_per_epoch}, dataset_len={dataset_len}). "
            "Check input paths/filters; TPU runs cannot proceed without batches."
        )
    # Keep an explicit attribute for downstream logging regardless of sharding mode
    try:
        train_loader.steps_per_epoch = steps_per_epoch
    except Exception:
        pass

    sched = setup_scheduler(opt, train_cfg, steps_per_epoch)

    for epoch in range(train_cfg.epochs):
        # No manual sampler set_epoch needed; Accelerator handles it if it wrapped the loader

        if accelerator.is_main_process:
            loader_iter = tqdm(
                train_loader,
                desc=f"E{epoch+1}",
                total=steps_per_epoch,
                disable=not accelerator.is_local_main_process
            )
        else:
            loader_iter = train_loader
            
        for step, batch in enumerate(loader_iter):
            if not shard_dataloaders:
                batch = {k: (v.to(accelerator.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            global_step = epoch * (steps_per_epoch or 0) + step
            
            batch_start = time.time()
            
            # Updated train step signature
            loss = train_step(model, batch, data_cfg, train_cfg, opt, sched, step, global_step, accelerator, use_xla_native)
            
            total_step_time = time.time() - batch_start

            if accelerator.is_main_process:
                # VRAM stats might be GPU specific, but let's keep basic logging
                current_lr = sched.get_last_lr()[0]
                
                if isinstance(loader_iter, tqdm):
                    loader_iter.set_postfix({
                        'loss': f"{loss:.4f}",
                        'step_time': f"{total_step_time:.1f}s"
                    })
                
                # Log training metrics
                wandb.log({
                    'train_loss': loss,
                    'learning_rate': current_lr,
                }, step=global_step)

            # Eval demo generation
            eval_every = getattr(args, 'eval_every_step', 0)
            if eval_every > 0 and global_step > 0 and global_step % eval_every == 0 and dac_model is not None:
                logger.info(f"[DEBUG] Eval trigger: eval_every={eval_every}, global_step={global_step}, dac_model={'present' if dac_model is not None else 'None'}")
                logger.info(f"[DEBUG] Waiting for all processes...")
                accelerator.wait_for_everyone()
                logger.info(f"[DEBUG] Setting model to eval mode...")
                model.eval()
                logger.info(f"[DEBUG] Entering torch.no_grad() context...")
                with torch.no_grad():
                    logger.info(f"[DEBUG] Calling eval_demo_step...")
                    eval_demo_step(
                        model=model,
                        data_cfg=data_cfg,
                        dac_model=dac_model,
                        global_step=global_step,
                        device=accelerator.device,
                        accelerator=accelerator,
                        train_cfg=train_cfg,
                    )
                logger.info(f"[DEBUG] Eval complete, setting model back to train mode...")
                model.train()
                logger.info(f"[DEBUG] Model back to train mode")

            should_save = train_cfg.save_step > 0 and global_step > 0 and (global_step % train_cfg.save_step == 0)
            if should_save:
                # Wait for everyone before saving
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ckpt = train_cfg.output_dir / f"ckpt_step{global_step}.pth"
                    state_dict = {k: v.cpu() for k, v in accelerator.get_state_dict(model).items()}
                    accelerator.save(state_dict, ckpt)
                    logger.info(f"Saved checkpoint: {ckpt}")
        if accelerator.is_main_process and (epoch + 1) == train_cfg.epochs:
            accelerator.wait_for_everyone()
            latest_ckpt = train_cfg.output_dir / "latest.pth"
            state_dict = {k: v.cpu() for k, v in accelerator.get_state_dict(model).items()}
            accelerator.save(state_dict, latest_ckpt)
            logger.info(f"Saved latest checkpoint: {latest_ckpt}")
    
    accelerator.wait_for_everyone()


def train_step(model, batch, data_cfg, train_cfg, opt, sched, step, global_step, accelerator: Accelerator, use_xla_native: bool):
    global _nonfinite_hits

    gen_val = ((global_step * 997 + train_cfg.seed) % 10000) / 10000.0
    if gen_val < train_cfg.unconditional_frac:
        pad_tok = data_cfg.text_pad_value
        batch['src_tokens'].fill_(pad_tok)
        batch['enc_self_attn_mask'] = torch.zeros_like(batch['enc_self_attn_mask'])
        batch['dec_cross_attn_mask'] = torch.zeros_like(batch['dec_cross_attn_mask'])

    # Manual gradient accumulation (like original Nari Labs script)
    # Don't use accelerator.accumulate() - it causes graph fragmentation on TPU
    with accelerator.autocast():
        logits = model(
            src_BxS=batch['src_tokens'],
            tgt_BxTxC=batch['tgt_tokens'],
            src_positions=batch['src_positions'],
            tgt_positions=batch['tgt_positions'],
            enc_self_attn_mask=batch['enc_self_attn_mask'],
            dec_self_attn_mask=batch['dec_self_attn_mask'],
            dec_cross_attn_mask=batch['dec_cross_attn_mask'],
            enable_dropout=False,
        )
        # Note: tgt_lens not used for TPU - we rely on audio_token_mask and ignore_index
        
        logits = logits[:, :-1]
        target = batch['tgt_tokens'][:, 1:, :]
        
        B, Tm1, C = target.shape
        pad_val = data_cfg.audio_pad_value
        
        # For TPU: Use fixed-shape operations only.
        # Don't use tgt_lens for masking - let ignore_index handle padding.
        # The audio_token_mask will exclude special tokens (BOS, EOS, PAD).
        
        channel_weights = []
        num_groups = C // 9
        if num_groups > 0:
            for _ in range(num_groups):
                channel_weights.extend(CODEBOOK_WEIGHTS)
        else:
            channel_weights = [1.0] * C

        loss_c = 0.0
        _, _, _, V = logits.size()
        
        # Only compute loss on valid audio tokens (0-1023), exclude special tokens
        audio_token_mask = (target >= 0) & (target <= 1023)
        
        # For TPU: Set non-audio tokens to pad_val so cross_entropy ignores them
        target_masked = torch.where(audio_token_mask, target, torch.full_like(target, pad_val))
        
        for c, w in enumerate(channel_weights):
            lc = logits[:, :, c, :].reshape(-1, V)
            tc = target_masked[:, :, c].reshape(-1)
            loss_c += w * F.cross_entropy(
                lc, tc,
                ignore_index=pad_val
            )
        loss = loss_c / sum(channel_weights)

    # Scale loss for gradient accumulation
    loss = loss / train_cfg.grad_accum_steps

    loss_detached = loss.detach()
    if not torch.isfinite(loss_detached):
        _nonfinite_hits += 1
        logger.warning(
            f"Non-finite loss at step {global_step} (hit {_nonfinite_hits}/{NONFINITE_HIT_LIMIT}); skipping backward"
        )
        opt.zero_grad()
        if HAS_XLA:
            xm.mark_step()
        if _nonfinite_hits >= NONFINITE_HIT_LIMIT:
            raise RuntimeError(f"Aborting: non-finite loss encountered {NONFINITE_HIT_LIMIT} times.")
        return float('nan')

    # Backward pass
    if use_xla_native and HAS_XLA:
        loss.backward()
    else:
        accelerator.backward(loss)
    
    # Manual gradient accumulation: only step optimizer every grad_accum_steps
    if (step + 1) % train_cfg.grad_accum_steps == 0:
        if use_xla_native and HAS_XLA:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            xm.optimizer_step(opt, barrier=False)
        else:
            accelerator.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
        sched.step()
        opt.zero_grad()
    
    # CRITICAL for TPU: mark_step tells XLA to execute the accumulated graph
    if HAS_XLA:
        xm.mark_step()

    # Return loss value
    return float(loss_detached) * train_cfg.grad_accum_steps


def run_training(args):
    # Initialize Accelerator
    # For TPU: Don't pass gradient_accumulation_steps - we handle it manually
    # to avoid graph fragmentation from accelerator.accumulate()
    accelerator = Accelerator(
        mixed_precision="bf16" if args.half else "no"
    )
    
    # Set seed
    seed_everything(args.seed)
    
    # Optional: improve bf16 throughput on Ampere+ GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    
    if accelerator.is_main_process:
        logger.info(f"Accelerator initialized. Device: {accelerator.device}, Num processes: {accelerator.num_processes}")

    delay_default = DataSettings().delay_pattern
    delay_pattern = _parse_delay_pattern(args.delay_pattern, delay_default)
    if len(delay_pattern) != args.channels:
        logger.warning(
            "delay_pattern length (%d) does not match channels (%d); pattern will be broadcast/truncated as needed.",
            len(delay_pattern),
            args.channels,
        )
    data_cfg = DataSettings(
        text_length=args.text_length,
        audio_length=args.audio_length,
        channels=args.channels,
        text_pad_value=args.text_pad_value,
        audio_eos_value=args.audio_eos_value,
        audio_pad_value=args.audio_pad_value,
        audio_bos_value=args.audio_bos_value,
        delay_pattern=delay_pattern,
    )
    model_cfg = ModelSettings(
        src_vocab_size=args.src_vocab_size,
        tgt_vocab_size=args.tgt_vocab_size,
        model_dim=args.model_dim,
        dropout=args.dropout,
    )

    use_sliding_window = not args.no_sliding_window

    dac_model = None
    need_dac_for_eval = args.eval_every_step > 0
    
    if args.preencoded_dir:
        dataset = PreEncodedDACDataset(args.preencoded_dir, data_cfg, use_sliding_window=use_sliding_window)
        # Load DAC for eval demos even with pre-encoded data
        if need_dac_for_eval:
            if accelerator.is_main_process:
                print("Loading DAC model for eval demos...")
            with accelerator.main_process_first():
                dac_ckpt = dac.utils.download()
            dac_model = dac.DAC.load(dac_ckpt).eval()
            removed_dac_wn = strip_weight_norms(dac_model)
            dac_model = dac_model.to(accelerator.device)
            if accelerator.is_main_process and removed_dac_wn:
                logger.info("Removed %d weight_norm wrappers from DAC model for XLA compatibility", removed_dac_wn)
    elif args.audio_folder:
        if accelerator.is_main_process:
            print("Loading DAC model...")
        with accelerator.main_process_first():
            dac_ckpt = dac.utils.download()
        dac_model = dac.DAC.load(dac_ckpt).eval()
        removed_dac_wn = strip_weight_norms(dac_model)
        dac_model = dac_model.to(accelerator.device)
        if accelerator.is_main_process and removed_dac_wn:
            logger.info("Removed %d weight_norm wrappers from DAC model for XLA compatibility", removed_dac_wn)
        dataset = AudioPromptDataset(args.audio_folder, data_cfg, dac_model, use_sliding_window=use_sliding_window)
    else:
        raise ValueError("Must specify either --audio_folder or --preencoded_dir")


    train_cfg = TrainConfig(
        epochs = args.epochs,
        batch_size = args.batch_size,
        grad_accum_steps = args.grad_accum_steps,
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
        warmup_steps = args.warmup_steps,
        unconditional_frac = args.unconditional_frac,
        run_name = args.run_name,
        output_dir = args.output_dir,
        seed = args.seed,
        no_decay_embed = args.no_decay_embed,
    )

    model = SimpleAudioModel(model_cfg, data_cfg)

    removed_model_wn = strip_weight_norms(model)
    if accelerator.is_main_process and removed_model_wn:
        logger.info("Removed %d weight_norm wrappers from model for XLA compatibility", removed_model_wn)

    if args.compile:
        if torch.cuda.is_available() and not HAS_XLA:
            model = torch.compile(model, backend="inductor")
        else:
            if accelerator.is_main_process:
                logger.warning("Skipping --compile: torch.compile(inductor) is only supported on CUDA (not TPU/XLA).")
    
    accelerator.wait_for_everyone()
    
    # Launch training
    train(model, data_cfg, dataset, train_cfg, args, accelerator, dac_model=dac_model)


def main():
    args = get_args()
    run_training(args)


if __name__ == "__main__":
    main()
