import argparse
import os
import random
import time


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN (if used)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    print("Starting CLI...")
    parser = argparse.ArgumentParser(description="Generate audio using the Dia model.")

    parser.add_argument("text", type=str, help="Input text for speech generation.")
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save the generated audio file (e.g., output.wav)."
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default="nari-labs/Dia-1.6B",
        help="Hugging Face repository ID (e.g., nari-labs/Dia-1.6B).",
    )
    parser.add_argument(
        "--local-paths", action="store_true", help="Load model from local config and checkpoint files."
    )

    parser.add_argument(
        "--config", type=str, help="Path to local config.json file (required if --local-paths is set)."
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to local model checkpoint .pth file (required if --local-paths is set)."
    )
    parser.add_argument(
        "--audio-prompt", type=str, default=None, help="Path to an optional audio prompt WAV file for voice cloning."
    )

    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument(
        "--max-tokens", type=int, default=None, help="Max tokens to generate (default: use config audio_length). Use small value like 10 for quick test."
    )
    gen_group.add_argument(
        "--cfg-scale", type=float, default=3.0, help="Classifier-Free Guidance scale (default: 3.0)."
    )
    gen_group.add_argument(
        "--temperature", type=float, default=1.3, help="Sampling temperature (higher is more random, default: 0.7)."
    )
    gen_group.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling probability (default: 0.95).")

    infra_group = parser.add_argument_group("Infrastructure")
    infra_group.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    infra_group.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on (e.g., 'cuda', 'cpu', 'tpu', default: auto).",
    )


    args = parser.parse_args()

    print("Importing numpy...", flush=True)
    import numpy as np
    print("Importing soundfile...", flush=True)
    import soundfile as sf
    print("Importing torch...", flush=True)
    import torch

    try:
        print("Importing torch_xla...", flush=True)
        import torch_xla.core.xla_model as xm
    except Exception:
        xm = None

    print("Importing Dia...", flush=True)
    from dia.model import Dia
    print("Dia import complete.", flush=True)

    # Validation for local paths
    if args.local_paths:
        if not args.config:
            parser.error("--config is required when --local-paths is set.")
        if not args.checkpoint:
            parser.error("--checkpoint is required when --local-paths is set.")
        if not os.path.exists(args.config):
            parser.error(f"Config file not found: {args.config}")
        if not os.path.exists(args.checkpoint):
            parser.error(f"Checkpoint file not found: {args.checkpoint}")

    # Set seed if provided
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Determine device
    print("Determining device...", flush=True)
    t_start = time.time()
    device_arg = args.device.lower()
    print(f"Device arg: {device_arg}", flush=True)
    
    if device_arg == "auto":
        # For inference CLI, default to CPU to avoid TPU multi-process coordination issues
        # TPU inference requires proper XLA setup (use Accelerate for that)
        if torch.cuda.is_available():
            device_arg = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_arg = "mps"
        else:
            device_arg = "cpu"
    print(f"Resolved device_arg: {device_arg}", flush=True)

    if device_arg in {"xla", "tpu"}:
        if xm is None:
            print("torch_xla is not available. Install it or use --device cpu.", flush=True)
            exit(1)
        # For single-process TPU inference, we need special setup
        # Use PJRT runtime with explicit device specification
        print("Setting up single-device TPU inference...", flush=True)
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        # Get the first TPU core (index 0)
        print("Calling xm.xla_device()...", flush=True)
        device = xm.xla_device()
        print(f"Got XLA device: {device}", flush=True)
    else:
        device = torch.device(device_arg)
    print(f"Using device: {device}", flush=True)

    # Load model
    print("Loading model...", flush=True)
    if args.local_paths:
        print(f"Loading from local paths: config='{args.config}', checkpoint='{args.checkpoint}'", flush=True)
        try:
            t_load = time.time()
            model = Dia.from_local(args.config, args.checkpoint, device=device)
            print(f"Model load time: {time.time() - t_load:.2f}s", flush=True)
        except Exception as e:
            print(f"Error loading local model: {e}", flush=True)
            import traceback
            traceback.print_exc()
            exit(1)
    else:
        print(f"Loading from Hugging Face Hub: repo_id='{args.repo_id}'", flush=True)
        try:
            t_load = time.time()
            model = Dia.from_pretrained(args.repo_id, device=device)
            print(f"Model load time: {time.time() - t_load:.2f}s", flush=True)
        except Exception as e:
            print(f"Error loading model from Hub: {e}", flush=True)
            import traceback
            traceback.print_exc()
            exit(1)
    print("Model loaded.", flush=True)

    # Generate audio
    print("Generating audio...", flush=True)
    try:
        sample_rate = 44100  # Default assumption

        t_gen = time.time()
        
        # Override max_tokens if specified (for quick testing)
        gen_kwargs = dict(
            text=args.text,
            audio_prompt_path=args.audio_prompt,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        if args.max_tokens:
            # Temporarily override config's audio_length
            original_audio_length = model.config.data.audio_length
            model.config.data.audio_length = args.max_tokens
            print(f"Overriding audio_length: {original_audio_length} -> {args.max_tokens}", flush=True)
        
        output_audio = model.generate(**gen_kwargs)
        
        if args.max_tokens:
            model.config.data.audio_length = original_audio_length
        print(f"Generation time: {time.time() - t_gen:.2f}s", flush=True)
        print("Audio generation complete.", flush=True)

        print(f"Saving audio to {args.output}...", flush=True)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

        # Ensure correct shape (frames, channels) and dtype for soundfile
        arr = output_audio
        if torch.is_tensor(arr):
            arr = arr.detach().cpu().numpy()
        if isinstance(arr, np.ndarray):
            if arr.ndim == 2 and arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
                arr = arr.T  # (channels, frames) -> (frames, channels)
            # Coerce to a common audio dtype if needed
            if arr.dtype not in (np.float32, np.float64, np.int16, np.int32):
                arr = arr.astype(np.float32)

        t_save = time.time()
        sf.write(args.output, arr, sample_rate)
        print(f"Save time: {time.time() - t_save:.2f}s", flush=True)
        print(f"Audio successfully saved to {args.output}", flush=True)

    except Exception as e:
        print(f"Error during audio generation or saving: {e}", flush=True)
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
