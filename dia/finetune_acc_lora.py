import argparse
import math
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from huggingface_hub import hf_hub_download

import dac

from .config import DiaConfig
from .layers import DiaModel, DenseGeneral, Attention, MlpBlock
from .dataset import MusicDataset, PreEncodedDACDataset
from . import finetune_acc

# Reuse training utilities from finetune_acc
from .finetune_acc import (
    setup_ddp,
    cleanup_ddp,
    TrainConfig,
    train,
)


class LoRAWrappedDenseGeneral1D(nn.Module):
    """LoRA wrapper for DenseGeneral layers that contract axis (-1,).

    Implements y = base(x) + (alpha/r) * ((x @ A) @ B) reshaped to base out_features.
    """

    def __init__(self, base: DenseGeneral, r: int, alpha: int, dropout: float):
        super().__init__()
        assert tuple(base.axis) == (-1,), "LoRAWrappedDenseGeneral1D expects axis=(-1,)"
        self.base = base
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        in_shapes = base.in_shapes  # (F_in,)
        out_features = base.out_features  # tuple like (heads, head_dim) or (num_act, hid)
        assert len(in_shapes) == 1, "LoRAWrappedDenseGeneral1D expects single contracted dim"
        self.in_features = int(in_shapes[0])
        self.out_features = tuple(int(d) for d in out_features)
        out_flat = 1
        for d in self.out_features:
            out_flat *= d
        self.out_flat = int(out_flat)

        p_dtype = getattr(base.weight, "dtype", torch.float32)
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, self.r, dtype=p_dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.r, self.out_flat, dtype=p_dtype))
        # Init: A ~ Kaiming, B zeros (standard LoRA)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_dtype = x.dtype
        x2 = self.dropout(x)
        x_flat = x2.reshape(-1, x2.shape[-1])  # (..., F_in)
        # Cast LoRA weights to match input dtype for mixed precision compatibility
        A = self.lora_A.to(x_dtype)
        B = self.lora_B.to(x_dtype)
        delta_flat = (x_flat @ A) @ B  # (..., out_flat)
        delta = delta_flat.view(*x2.shape[:-1], *self.out_features)
        return (base_out + self.scaling * delta).to(x_dtype)


class LoRAWrappedDenseGeneral2D(nn.Module):
    """LoRA wrapper for DenseGeneral layers that contract axis (-2,-1)."""

    def __init__(self, base: DenseGeneral, r: int, alpha: int, dropout: float):
        super().__init__()
        assert tuple(base.axis) == (-2, -1), "LoRAWrappedDenseGeneral2D expects axis=(-2,-1)"
        self.base = base
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        in_shapes = base.in_shapes  # (N, H)
        out_features = base.out_features  # (D_out,)
        assert len(in_shapes) == 2 and len(out_features) == 1, "LoRA2D expects 2D in_shapes and 1D out_features"
        self.in_features = int(in_shapes[0]) * int(in_shapes[1])
        self.out_features = int(out_features[0])

        p_dtype = getattr(base.weight, "dtype", torch.float32)
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, self.r, dtype=p_dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.r, self.out_features, dtype=p_dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_dtype = x.dtype
        x2 = self.dropout(x)
        prefix = x2.shape[:-2]
        x_flat = x2.reshape(-1, x2.shape[-2] * x2.shape[-1])  # (..., N*H)
        # Cast LoRA weights to match input dtype for mixed precision compatibility
        A = self.lora_A.to(x_dtype)
        B = self.lora_B.to(x_dtype)
        delta_flat = (x_flat @ A) @ B  # (..., D_out)
        delta = delta_flat.view(*prefix, self.out_features)
        return (base_out + self.scaling * delta).to(x_dtype)


def inject_lora_into_model(model: nn.Module, r: int, alpha: int, dropout: float, targets: set[str]):
    """Replace target DenseGeneral projections with LoRA-wrapped modules.

    targets subset: {"attn_qkv", "attn_o", "mlp", "logits"}
    """
    for mod in model.modules():
        # Attention q/k/v with axis=(-1,)
        if isinstance(mod, Attention) and "attn_qkv" in targets:
            mod.q_proj = LoRAWrappedDenseGeneral1D(mod.q_proj, r, alpha, dropout)
            mod.k_proj = LoRAWrappedDenseGeneral1D(mod.k_proj, r, alpha, dropout)
            mod.v_proj = LoRAWrappedDenseGeneral1D(mod.v_proj, r, alpha, dropout)
        # Attention o with axis=(-2,-1)
        if isinstance(mod, Attention) and "attn_o" in targets:
            mod.o_proj = LoRAWrappedDenseGeneral2D(mod.o_proj, r, alpha, dropout)
        # MLP fused in/out
        if isinstance(mod, MlpBlock) and "mlp" in targets:
            mod.wi_fused = LoRAWrappedDenseGeneral1D(mod.wi_fused, r, alpha, dropout)
            mod.wo = LoRAWrappedDenseGeneral1D(mod.wo, r, alpha, dropout)

    # Decoder logits projection
    if "logits" in targets:
        dec = getattr(model, "decoder", None)
        if dec is not None and hasattr(dec, "logits_dense"):
            dec.logits_dense = LoRAWrappedDenseGeneral1D(dec.logits_dense, r, alpha, dropout)


def freeze_non_lora_params(model: nn.Module):
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False


def count_trainable(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def save_adapter_only(model: nn.Module, path: Path):
    sd = model.state_dict()
    adapter_sd = {k: v for k, v in sd.items() if "lora_" in k}
    torch.save(adapter_sd, path)


def get_lora_cli_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_targets",
        type=str,
        default="attn_qkv,attn_o",
        help="Comma-separated subset of: attn_qkv,attn_o,mlp,logits",
    )
    p.add_argument(
        "--adapter_only_ckpt",
        action="store_true",
        help="Save adapter-only checkpoint (keys containing 'lora_') at end of training",
    )
    # Return both parsed LoRA args and leftover argv for the base parser
    return p.parse_known_args(sys.argv[1:])


def run_ddp_worker_lora(rank: int, world_size: int, args):
    try:
        # Setup DDP
        setup_ddp(rank, world_size)

        # Determinism
        torch.set_float32_matmul_precision("high")

        device = torch.device(f"cuda:{rank}")

        # Load config
        dia_cfg = DiaConfig.load(args.config)

        # Configure tag augmentation in the finetune_acc module
        finetune_acc.TAG_SHUFFLE = False if getattr(args, 'tag_no_shuffle', False) else bool(getattr(args, 'tag_shuffle', True))
        finetune_acc.TAG_DROPOUT = float(getattr(args, 'tag_dropout', 0.0))
        finetune_acc.TAG_LIMIT = getattr(args, 'tag_limit', None)

        # DAC model (on device, must be in eval mode for deterministic encoding)
        dac_model = dac.DAC.load(dac.utils.download()).eval().to(device)

        # Dataset
        use_sliding_window = args.use_sliding_window
        if args.preencoded_dir:
            dataset = PreEncodedDACDataset(args.preencoded_dir, dia_cfg, use_sliding_window)
        elif args.audio_folder:
            skip_tags_list = [t.strip() for t in args.skip_tags.split(',')] if args.skip_tags else None
            dataset = MusicDataset(args.audio_folder, dia_cfg, dac_model, use_sliding_window,
                                 ignore_missing_prompts=not args.require_prompts,
                                 skip_tags=skip_tags_list)
        else:
            raise ValueError("Must specify either --audio_folder or --preencoded_dir")

        # Train config
        train_cfg = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            unconditional_frac=args.unconditional_frac,
            eval_step=args.eval_step,
            run_name=args.run_name or TrainConfig.run_name,
            output_dir=args.output_dir,
            seed=args.seed,
            no_decay_embed=args.no_decay_embed,
        )

        # Model
        ckpt_file = args.local_ckpt or hf_hub_download(args.hub_model, filename="dia-v0_1.pth")
        model = DiaModel(dia_cfg)
        state = torch.load(ckpt_file, map_location="cpu")
        # Track if we performed expansion to know if we should warm-start stereo
        expanded_stereo = False
        
        # Adapt checkpoint for stereo (9 -> 18 channels) before loading to avoid size mismatch
        try:
            key = "decoder.logits_dense.weight"
            if key in state:
                expected_W = model.decoder.logits_dense.weight
                W_ckpt = state[key]
                if (
                    W_ckpt is not None
                    and expected_W is not None
                    and W_ckpt.dim() == 3 and expected_W.dim() == 3
                    and W_ckpt.shape[0] == expected_W.shape[0]
                    and W_ckpt.shape[2] == expected_W.shape[2]
                    and W_ckpt.shape[1] != expected_W.shape[1]
                ):
                    if W_ckpt.shape[1] * 2 == expected_W.shape[1]:
                        state[key] = torch.cat([W_ckpt, W_ckpt], dim=1)
                        expanded_stereo = True
                        print(f"Expanded {key} from {tuple(W_ckpt.shape)} to {tuple(state[key].shape)} by duplication")
                    else:
                        del state[key]
                        print(f"Removed {key} due to incompatible shape {tuple(W_ckpt.shape)} -> expected {tuple(expected_W.shape)}")
        except Exception as e:
            print(f"While adapting checkpoint weights: {e}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint with strict=False; missing={len(missing)}, unexpected={len(unexpected)}")
        
        # Warm-start stereo by duplicating left-channel params into right, ONLY if expanding 9->18
        # If we loaded an existing stereo checkpoint, expanded_stereo is False, so we skip this
        if expanded_stereo:
            try:
                if dia_cfg.data.channels == 18:
                    print("Warm-starting stereo channels (copying Left -> Right)...")
                    if hasattr(model.decoder, "embeddings") and len(model.decoder.embeddings) >= 18:
                        for i in range(9, 18):
                            model.decoder.embeddings[i].weight.data.copy_(model.decoder.embeddings[i - 9].weight.data)
                    W = model.decoder.logits_dense.weight  # (E, C, V)
                    if W.dim() == 3 and W.shape[1] >= 18:
                        W.data[:, 9:18, :].copy_(W.data[:, 0:9, :])
            except Exception as e:
                print(f"Stereo warm-start duplication skipped: {e}")
        else:
            if dia_cfg.data.channels == 18:
                print("Skipping stereo warm-start duplication (loaded checkpoint appears to be stereo already)")

        # Dtype consistency first
        if args.half:
            model = model.half()
            for p in model.parameters():
                if p.dtype != torch.float16:
                    p.data = p.data.half()
        else:
            for p in model.parameters():
                if p.dtype != torch.float32:
                    p.data = p.data.float()

        # Inject LoRA on CPU, then move/model.to(rank) happens inside train()
        targets = set(t.strip() for t in args.lora_targets.split(",") if t.strip())
        inject_lora_into_model(model, args.lora_r, args.lora_alpha, args.lora_dropout, targets)

        # Reload state dict to capture any pre-existing LoRA weights that were ignored before injection
        # (e.g. if resuming from a LoRA checkpoint)
        if any("lora_" in k for k in state.keys()):
            print("Detected LoRA weights in checkpoint, reloading after injection...")
            missing_lora, unexpected_lora = model.load_state_dict(state, strict=False)
            print(f"Reloaded checkpoint after LoRA injection; missing={len(missing_lora)}")

        freeze_non_lora_params(model)

        # Barrier before training
        torch.distributed.barrier()

        # Train (DDP inside train())
        train(model, dia_cfg, dac_model, dataset, train_cfg, args, rank=rank, world_size=world_size, use_ddp=True)

        # Save adapter-only on rank 0
        if rank == 0 and getattr(args, "adapter_only_ckpt", False):
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            save_adapter_only(model.module if hasattr(model, "module") else model, out / "lora_adapter_only.pth")

    except Exception as e:
        print(f"Error in LoRA DDP worker rank {rank}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_ddp()


def main():
    # Parse base args by reusing finetune_acc.get_args()
    from .finetune_acc import get_args as base_get_args

    # Parse LoRA args first and strip them from argv so base parser won't see them
    lora_args, remaining_argv = get_lora_cli_args()

    orig_argv = sys.argv
    try:
        sys.argv = [orig_argv[0]] + remaining_argv
        args = base_get_args()
    finally:
        sys.argv = orig_argv

    # Attach LoRA args onto base args
    args.lora_r = lora_args.lora_r
    args.lora_alpha = lora_args.lora_alpha
    args.lora_dropout = lora_args.lora_dropout
    args.lora_targets = lora_args.lora_targets
    args.adapter_only_ckpt = lora_args.adapter_only_ckpt

    world_size = torch.cuda.device_count()

    # Single GPU or forced single-GPU
    if world_size < 2 or args.force_single_gpu:
        if args.force_single_gpu:
            print("Forcing single GPU training as requested")
        else:
            print("WARNING: Multi-GPU requested but only one GPU available; running single GPU")

        torch.set_float32_matmul_precision("high")

        dia_cfg = DiaConfig.load(args.config)

        # Configure tag augmentation for single process
        finetune_acc.TAG_SHUFFLE = False if getattr(args, 'tag_no_shuffle', False) else bool(getattr(args, 'tag_shuffle', True))
        finetune_acc.TAG_DROPOUT = float(getattr(args, 'tag_dropout', 0.0))
        finetune_acc.TAG_LIMIT = getattr(args, 'tag_limit', None)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dac_model = dac.DAC.load(dac.utils.download()).eval().to(device)

        # Dataset
        use_sliding_window = args.use_sliding_window
        if args.preencoded_dir:
            dataset = PreEncodedDACDataset(args.preencoded_dir, dia_cfg, use_sliding_window)
        elif args.audio_folder:
            skip_tags_list = [t.strip() for t in args.skip_tags.split(',')] if args.skip_tags else None
            dataset = MusicDataset(args.audio_folder, dia_cfg, dac_model, use_sliding_window,
                                 ignore_missing_prompts=not args.require_prompts,
                                 skip_tags=skip_tags_list)
        else:
            raise ValueError("Must specify either --audio_folder or --preencoded_dir")

        # Train config
        train_cfg = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            unconditional_frac=args.unconditional_frac,
            eval_step=args.eval_step,
            run_name=args.run_name or TrainConfig.run_name,
            output_dir=args.output_dir,
            seed=args.seed,
            no_decay_embed=args.no_decay_embed,
        )

        # Model
        ckpt_file = args.local_ckpt or hf_hub_download(args.hub_model, filename="dia-v0_1.pth")
        model = DiaModel(dia_cfg)
        state = torch.load(ckpt_file, map_location="cpu")
        # Track if we performed expansion to know if we should warm-start stereo
        expanded_stereo = False
        
        # Adapt checkpoint for stereo (9 -> 18 channels) before loading to avoid size mismatch
        try:
            key = "decoder.logits_dense.weight"
            if key in state:
                expected_W = model.decoder.logits_dense.weight
                W_ckpt = state[key]
                if (
                    W_ckpt is not None
                    and expected_W is not None
                    and W_ckpt.dim() == 3 and expected_W.dim() == 3
                    and W_ckpt.shape[0] == expected_W.shape[0]
                    and W_ckpt.shape[2] == expected_W.shape[2]
                    and W_ckpt.shape[1] != expected_W.shape[1]
                ):
                    if W_ckpt.shape[1] * 2 == expected_W.shape[1]:
                        state[key] = torch.cat([W_ckpt, W_ckpt], dim=1)
                        expanded_stereo = True
                        print(f"Expanded {key} from {tuple(W_ckpt.shape)} to {tuple(state[key].shape)} by duplication")
                    else:
                        del state[key]
                        print(f"Removed {key} due to incompatible shape {tuple(W_ckpt.shape)} -> expected {tuple(expected_W.shape)}")
        except Exception as e:
            print(f"While adapting checkpoint weights: {e}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint with strict=False; missing={len(missing)}, unexpected={len(unexpected)}")
        
        # Warm-start stereo by duplicating left-channel params into right, ONLY if expanding 9->18
        # If we loaded an existing stereo checkpoint, expanded_stereo is False, so we skip this
        if expanded_stereo:
            try:
                if dia_cfg.data.channels == 18:
                    print("Warm-starting stereo channels (copying Left -> Right)...")
                    if hasattr(model.decoder, "embeddings") and len(model.decoder.embeddings) >= 18:
                        for i in range(9, 18):
                            model.decoder.embeddings[i].weight.data.copy_(model.decoder.embeddings[i - 9].weight.data)
                    W = model.decoder.logits_dense.weight
                    if W.dim() == 3 and W.shape[1] >= 18:
                        W.data[:, 9:18, :].copy_(W.data[:, 0:9, :])
            except Exception as e:
                print(f"Stereo warm-start duplication skipped: {e}")
        else:
            if dia_cfg.data.channels == 18:
                print("Skipping stereo warm-start duplication (loaded checkpoint appears to be stereo already)")

        # Dtype consistency
        if args.half:
            model = model.half()
            for p in model.parameters():
                if p.dtype != torch.float16:
                    p.data = p.data.half()
        else:
            for p in model.parameters():
                if p.dtype != torch.float32:
                    p.data = p.data.float()

        # Inject LoRA and freeze
        targets = set(t.strip() for t in args.lora_targets.split(",") if t.strip())
        inject_lora_into_model(model, args.lora_r, args.lora_alpha, args.lora_dropout, targets)

        # Reload state dict to capture any pre-existing LoRA weights that were ignored before injection
        if any("lora_" in k for k in state.keys()):
            print("Detected LoRA weights in checkpoint, reloading after injection...")
            missing_lora, unexpected_lora = model.load_state_dict(state, strict=False)
            print(f"Reloaded checkpoint after LoRA injection; missing={len(missing_lora)}")

        freeze_non_lora_params(model)

        # Report trainable count
        t_tr, t_tot = count_trainable(model)
        print(f"Trainable params: {t_tr:,} / {t_tot:,} ({100.0 * t_tr / max(1, t_tot):.4f}%)")

        # Train
        train(model, dia_cfg, dac_model, dataset, train_cfg, args, rank=0, world_size=1, use_ddp=False)

        # Save adapter-only if requested
        if getattr(args, "adapter_only_ckpt", False):
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            save_adapter_only(model, out / "lora_adapter_only.pth")

        return

    # Multi-GPU DDP branch
    print(f"Launching LoRA DDP training with {world_size} processes...")
    mp.set_start_method("spawn", force=True)
    try:
        mp.spawn(
            run_ddp_worker_lora,
            args=(world_size, args),
            nprocs=world_size,
            join=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()


