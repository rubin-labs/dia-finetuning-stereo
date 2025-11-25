#!/usr/bin/env python3
import argparse
import os
import json
import torch
import soundfile as sf
import numpy as np
from dia.model import Dia
from dia.config import DiaConfig

# Prompts provided by the user
PROMPTS = {
  "nm_canals 140bpm.pt": {
    "text": "new age, piano, strings, bass, peaceful, calm, reflective, instrumental",
    "sr": 44100,
    "channels": 18,
    "length": 2363
  },
  "nm_boo 130bpm.pt": {
    "text": "piano, strings, organ, classical, new age, calm, peaceful, reflective",
    "sr": 44100,
    "channels": 18,
    "length": 3817
  },
  "nm_nomansland 155bpm.pt": {
    "text": "electronic, trance, energetic, driving, uplifting, synthesizer",
    "sr": 44100,
    "channels": 18,
    "length": 3201
  },
  "nm_wins 130bpm.pt": {
    "text": "orchestral, classical, cinematic, calm, emotional, piano, strings, synth, woodwinds",
    "sr": 44100,
    "channels": 18,
    "length": 2545
  },
  "nm_pairs v2 165bpm.pt": {
    "text": "electronic, dance, energetic, upbeat, driving, synthesizer",
    "sr": 44100,
    "channels": 18,
    "length": 1003
  },
  "nm_cracks 92bpm.pt": {
    "text": "lo-fi, chill-hop, chill, melancholic, atmospheric, piano, drums, bass, strings, bells",
    "sr": 44100,
    "channels": 18,
    "length": 1798
  },
  "nm_funnels 160bpm.pt": {
    "text": "cinematic, orchestral, mysterious, dramatic, atmospheric, strings, piano",
    "sr": 44100,
    "channels": 18,
    "length": 1034
  },
  "nm_canals v2 150bpm.pt": {
    "text": "ambient, new age, classical crossover, calm, peaceful, reflective, piano, strings, pads",
    "sr": 44100,
    "channels": 18,
    "length": 2205
  },
  "nm_martins v2 150bpm.pt": {
    "text": "electronic, techno, dark, energetic, synth",
    "sr": 44100,
    "channels": 18,
    "length": 1103
  },
  "nm_makeups v2 165bpm.pt": {
    "text": "electronic, dance, upbeat, driving, synthesizer",
    "sr": 44100,
    "channels": 18,
    "length": 1003
  },
  "nm_dumptruck v2 160bpm.pt": {
    "text": "chiptune, video game music, playful, upbeat, synthesizer",
    "sr": 44100,

    "channels": 18,
    "length": 1034
  },
  "nm_forlorn v2 140bpm.pt": {
    "text": "classical, ambient, piano, strings, sad, melancholic, somber, reflective",
    "sr": 44100,
    "channels": 18,
    "length": 2363
  },
  "nm_healy 155bpm.pt": {
    "text": "electronic, ambient, calm, ethereal, melancholic, bells, synth pad, synth bass",
    "sr": 44100,
    "channels": 18,
    "length": 1067
  },
  "nm_teases 170bpm.pt": {
    "text": "ambient, calm, melancholic, harp, synthesizer",
    "sr": 44100,
    "channels": 18,
    "length": 1946
  },
  "nm_foulplay 170bpm.pt": {
    "text": "electronic, dance, trance, energetic, driving, rhythmic, atmospheric, hypnotic, synthesizer",
    "sr": 44100,
    "channels": 18,
    "length": 1946
  },
  "nm_pairs 140bpm.pt": {
    "text": "pop, instrumental, piano, reflective, light",
    "sr": 44100,
    "channels": 18,
    "length": 3544
  },
  "nm_martins 80bpm.pt": {
    "text": "synthwave, chillwave, lo-fi, chill, dreamlike, atmospheric, synthesizer",
    "sr": 44100,
    "channels": 18,
    "length": 2068
  },
  "nm_wins v2 160bpm.pt": {
    "text": "piano, strings, light percussion, instrumental, melancholic, peaceful, reflective, dreamy",
    "sr": 44100,
    "channels": 18,
    "length": 1034
  },
  "nm_makeups 140bpm.pt": {
    "text": "electronic, chiptune, energetic, synth, arpeggiator",
    "sr": 44100,
    "channels": 18,
    "length": 1182
  },
  "nm_sangria v2 160bpm.pt": {
    "text": "electronic, trance, dance, energetic, driving, uplifting, synthesizer",
    "sr": 44100,
    "channels": 18,
    "length": 1034
  },
  "nm_boo v2 160bpm.pt": {
    "text": "cinematic, suspenseful, dark, energetic, mysterious, strings, bells, bass",
    "sr": 44100,
    "channels": 18,
    "length": 1034
  },
  "nm_trish 140bpm.pt": {
    "text": "retro game music, whimsical, playful, dreamy, pizzicato strings, bells, synth pad",
    "sr": 44100,
    "channels": 18,
    "length": 2363
  },
  "nm_weeds 150bpm.pt": {
    "text": "orchestral, underscore, soundtrack, strings, piano, pizzicato, suspenseful, mysterious, playful, tense, fast",
    "sr": 44100,
    "channels": 18,
    "length": 2205
  },
  "nm_weeds v2 160bpm.pt": {
    "text": "classical, orchestral, cinematic, emotional, strings, piano",
    "sr": 44100,
    "channels": 18,
    "length": 2068
  },
  "nm_nomansland v2 140bpm.pt": {
    "text": "electronic, trance, energetic, driving, synth",
    "sr": 44100,
    "channels": 18,
    "length": 4725
  },
  "nm_sharktank 160bpm.pt": {
    "text": "instrumental, rock, energetic, driving, electric guitar",
    "sr": 44100,
    "channels": 18,
    "length": 2068
  },
  "nm_sangria 140bpm.pt": {
    "text": "new age, ambient, electronic, peaceful, dreamy, calm, relaxing, synthesizer, arpeggiator, bells",
    "sr": 44100,
    "channels": 18,
    "length": 4725
  },
  "nm_crooks 80bpm.pt": {
    "text": "piano, pads, ambient, cinematic, melancholic, peaceful, reflective, instrumental",
    "sr": 44100,
    "channels": 18,
    "length": 6202
  },
  "nm_growls 130bpm.pt": {
    "text": "traditional chinese, guzheng, plucked strings, calm, peaceful, meditative",
    "sr": 44100,
    "channels": 18,
    "length": 1273
  },
  "nm_capela 168bpm.pt": {
    "text": "ambient, cinematic, mysterious, melancholic, calm, pizzicato, strings",
    "sr": 44100,
    "channels": 18,
    "length": 2954
  },
  "nm_orders 155bpm.pt": {
    "text": "electronic, trance, energetic, synthesizer",
    "sr": 44100,
    "channels": 18,
    "length": 3201
  },
  "nm_equinox 160bpm.pt": {
    "text": "electronic, trance, energetic, fast, synth",
    "sr": 44100,
    "channels": 18,
    "length": 3101
  },
  "nm_dumptruck 140bpm.pt": {
    "text": "piano, strings, drums, neoclassical, cinematic, melancholic, thoughtful",
    "sr": 44100,
    "channels": 18,
    "length": 1182
  },
  "sample.pt": {
    "text": "electronic, dance, house, upbeat, energetic, groovy, synthesizer",
    "sr": 44100,
    "channels": 18,
    "length": 1134
  },
  "nm_bigmidbigguap 170bpm.pt": {
    "text": "piano, classical, ambient, instrumental, peaceful, melancholic",
    "sr": 44100,
    "channels": 18,
    "length": 973
  },
  "nm_dynamite 92bpm.pt": {
    "text": "piano, strings, ambient, cinematic, relaxing, peaceful, melancholic, reflective, new age",
    "sr": 44100,
    "channels": 18,
    "length": 7191
  },
  "nm_wess 150bpm.pt": {
    "text": "acoustic guitar, instrumental, melancholic, calm, somber",
    "sr": 44100,
    "channels": 18,
    "length": 2205
  },
  "nm_bananaclip 140bpm.pt": {
    "text": "instrumental, cinematic, calm, melancholy, strings, piano",
    "sr": 44100,
    "channels": 18,
    "length": 2363
  },
  "nm_baja 170bpm.pt": {
    "text": "electronic, trance, energetic, driving, synthesizer",
    "sr": 44100,
    "channels": 18,
    "length": 973
  },
  "nm_forlorn 160bpm.pt": {
    "text": "piano, strings, melancholic, sad, forlorn, somber, classical, cinematic, instrumental",
    "sr": 44100,
    "channels": 18,
    "length": 2068
  },
  "nm_bananaclip v2 160bpm.pt": {
    "text": "ambient, electronic, eerie, atmospheric, mysterious, dark, synthesizer",
    "sr": 44100,
    "channels": 18,
    "length": 2068
  }
}

# Inference parameter combinations to test
# Format: (cfg_scale, temperature, top_p)
PARAM_COMBINATIONS = [
    (3.0, 1.3, 0.95),   # Default
    (1.0, 0.8, 0.95),   # Low CFG, Low Temp (more focused)
    (5.0, 1.3, 0.95),   # High CFG
    (3.0, 1.0, 0.90),   # Default CFG, lower temp/top_p
]

import random

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
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
    parser = argparse.ArgumentParser(description="Test inference with various parameters.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--config", type=str, default="./dia/config_overfit_inference.json", help="Path to the model config.")
    parser.add_argument("--output_dir", type=str, default="inference_tests", help="Directory to save outputs.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Using device: {args.device}")
    print(f"Loading model from {args.checkpoint} with config {args.config}")
    
    # Load model
    try:
        model = Dia.from_local(args.config, args.checkpoint, device=torch.device(args.device))
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Iterate through prompts
    for name, data in PROMPTS.items():
        text_prompt = data["text"]
        base_name = os.path.splitext(name)[0]
        
        print(f"\nProcessing: {name}")
        print(f"Prompt: {text_prompt}")

        for cfg, temp, top_p in PARAM_COMBINATIONS:
            print(f"  Params: CFG={cfg}, Temp={temp}, TopP={top_p}")
            
            try:
                # Generate audio
                # Note: model.generate returns numpy array (channels, samples) or (samples, channels) depending on implementation
                # Inspecting model.py, it returns (channels, samples) for dac decoding but let's verify return of generate()
                # Dia.generate returns: audio.squeeze().cpu().numpy()
                # codebook_to_audio returns [B, C, T] ?
                # Let's just handle the output carefully.
                
                output_audio = model.generate(
                    text=text_prompt,
                    cfg_scale=cfg,
                    temperature=temp,
                    top_p=top_p,
                )
                
                # Construct filename
                filename = f"{base_name}_cfg{cfg}_temp{temp}_topP{top_p}.wav"
                filepath = os.path.join(args.output_dir, filename)
                
                # Save audio
                # Ensure correct shape for soundfile: (frames, channels)
                arr = output_audio
                if isinstance(arr, np.ndarray):
                     if arr.ndim == 2:
                        # If shape is (channels, frames), transpose to (frames, channels)
                        # Typically stereo is (2, N), so if dim 0 is small, it's channels.
                        if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 128: # 18 channels is common here
                             arr = arr.T
                
                sf.write(filepath, arr, 44100)
                print(f"    Saved to {filepath}")
                
            except Exception as e:
                print(f"    Error generating/saving: {e}")

if __name__ == "__main__":
    main()

