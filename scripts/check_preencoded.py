
import argparse
from pathlib import Path
import torch
import soundfile as sf
import dac
import json
from tqdm import tqdm

def decode_in_chunks(dac_model, z, chunk_length=500):
    """Decode latent vectors in chunks to avoid OOM."""
    # z shape: (1, 1024, T)
    T = z.shape[-1]
    audio_chunks = []
    
    print(f"Decoding {T} frames in chunks of {chunk_length}...")
    
    with torch.no_grad():
        for i in tqdm(range(0, T, chunk_length)):
            z_chunk = z[:, :, i:i+chunk_length]
            audio_chunk = dac_model.decode(z_chunk)
            audio_chunks.append(audio_chunk.cpu())
            
    return torch.cat(audio_chunks, dim=-1)

def check_preencoded_file():
    parser = argparse.ArgumentParser(description="Check a pre-encoded .pt file by decoding it with DAC")
    parser.add_argument("pt_file", type=Path, help="Path to .pt file")
    parser.add_argument("--output", type=Path, default="check_output.wav", help="Output wav file")
    args = parser.parse_args()

    print(f"Loading DAC model...")
    # Load DAC model
    model_path = dac.utils.download(model_type="44khz")
    dac_model = dac.DAC.load(model_path)
    dac_model.eval()
    
    # Use GPU if available
    if torch.cuda.is_available():
        print("Using GPU for decoding")
        dac_model = dac_model.cuda()
    else:
        print("Using CPU for decoding")

    print(f"Loading {args.pt_file}...")
    codes = torch.load(args.pt_file, map_location="cpu")
    
    # codes shape: usually (T, C) = (T, 9) or (T, 18)
    print(f"Codes shape: {codes.shape}")
    
    # Ensure codes are LongTensor for embedding lookup
    codes = codes.long()
    
    # Check values
    min_val = codes.min().item()
    max_val = codes.max().item()
    print(f"Value range: [{min_val}, {max_val}]")
    
    if max_val >= 1024:
        print("WARNING: Found values >= 1024. This is not a valid DAC codebook index!")
    
    # Prepare for decoding
    # DAC decode expects (B, C, T) = (1, 9/18, T)
    # codes is (T, C)
    codes = codes.transpose(0, 1).unsqueeze(0) # (1, C, T)
    
    # Explicitly handle Stereo (18 channels) vs Mono (9 channels)
    if codes.shape[1] == 18:
        print("Detected stereo codes (18). Splitting for decoding...")
        left_codes = codes[:, :9, :]
        right_codes = codes[:, 9:, :]
        
        if torch.cuda.is_available():
            left_codes = left_codes.cuda()
            right_codes = right_codes.cuda()
        
        # Convert to embeddings first
        with torch.no_grad():
            z_left, _, _ = dac_model.quantizer.from_codes(left_codes)
            z_right, _, _ = dac_model.quantizer.from_codes(right_codes)
        
        # Decode in chunks
        print("Decoding Left Channel...")
        audio_left = decode_in_chunks(dac_model, z_left)
        print("Decoding Right Channel...")
        audio_right = decode_in_chunks(dac_model, z_right)
        
        # Combine
        audio = torch.cat([audio_left, audio_right], dim=1) # (1, 2, T_samples)
        
        # Save
        audio = audio.squeeze(0).detach().cpu().numpy()
        if audio.shape[0] == 2:
            audio = audio.T # (T, 2)
        
        sf.write(args.output, audio, 44100)
        print(f"Saved to {args.output}")
        return

    # Mono case
    if torch.cuda.is_available():
        codes = codes.cuda()
        
    try:
        with torch.no_grad():
            z, _, _ = dac_model.quantizer.from_codes(codes)
            audio = decode_in_chunks(dac_model, z)
        
        # Save
        audio = audio.squeeze(0).detach().cpu().numpy()
        if audio.ndim == 2 and audio.shape[0] == 2:
             audio = audio.T
             
        sf.write(args.output, audio, 44100)
        print(f"Saved to {args.output}")

    except Exception as e:
        print(f"Failed to decode: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_preencoded_file()
