#!/usr/bin/env python3
import sys
import os
import json
import glob
import argparse
import subprocess
import torch
import numpy as np
import soundfile as sf
import pandas as pd
import librosa
from tqdm import tqdm
from typing import List, Dict, Optional

# Ensure we import from the local dia/ folder
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from dia.model import Dia

# Try importing optional dependencies
try:
    import laion_clap
    HAS_CLAP = True
except ImportError:
    HAS_CLAP = False

try:
    from frechet_audio_distance import FrechetAudioDistance
    HAS_FAD = True
except ImportError:
    HAS_FAD = False

# Standard Evaluation Prompts
EVAL_PROMPTS = {
    "calm_piano": "solo piano, ambient, peaceful, reverb, 80bpm",
    "electronic_dance": "edm, driving beat, synthesizer, 128bpm, club",
    "jazz_fusion": "jazz fusion, saxophone solo, complex chords, fast tempo",
    "orchestral_build": "cinematic, orchestra, crescendo, epic, movie score",
    "unseen_mix": "heavy metal mixed with bagpipes, aggressive, distortion",
    "progressive_structure": "song structure, intro, verse, chorus, bridge",
}
EVAL_SEEDS = [42]
DEFAULT_PARAMS = {"cfg_scale": 3.0, "temperature": 1.0, "top_p": 0.95}


# --- Metrics Classes ---

def compute_beat_stability(y: np.ndarray, sr: int) -> float:
    """
    Estimates beat stability by calculating the variance of inter-beat intervals.
    Lower variance = more stable tempo.
    Returns: Coefficient of Variation of IBIs (std/mean). Lower is better.
    """
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        if len(beat_frames) < 2:
            return 1.0 # Unstable / no beat
            
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        ibis = np.diff(beat_times)
        if len(ibis) < 1:
            return 1.0
            
        # Coefficient of variation: std / mean
        cv = np.std(ibis) / (np.mean(ibis) + 1e-6)
        return float(cv)
    except:
        return 1.0

def compute_key_consistency(y: np.ndarray, sr: int) -> float:
    """
    Estimates key consistency by correlating chroma vectors over time.
    Returns: Average correlation between adjacent segments. Higher (close to 1.0) = consistent key.
    """
    try:
        # Compute chroma (pitch class profile)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # Split into chunks (e.g. 4 chunks)
        n_chunks = 4
        chunk_len = chroma.shape[1] // n_chunks
        if chunk_len < 10: return 0.0
        
        correlations = []
        for i in range(n_chunks - 1):
            c1 = chroma[:, i*chunk_len : (i+1)*chunk_len].mean(axis=1)
            c2 = chroma[:, (i+1)*chunk_len : (i+2)*chunk_len].mean(axis=1)
            
            # Cosine similarity
            sim = np.dot(c1, c2) / (np.linalg.norm(c1)*np.linalg.norm(c2) + 1e-8)
            correlations.append(sim)
            
        return float(np.mean(correlations)) if correlations else 0.0
    except:
        return 0.0

def compute_audio_quality_metrics(audio_path: str) -> Dict[str, float]:
    """Computes basic signal quality metrics for a single audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # 1. Clipping Ratio
        clipping_threshold = 0.99
        num_clipped = np.sum(np.abs(y) >= clipping_threshold)
        clipping_ratio = num_clipped / len(y)
        
        # 2. Silence Ratio
        db = librosa.amplitude_to_db(np.abs(y), ref=np.max)
        silence_threshold_db = -60
        num_silent = np.sum(db < silence_threshold_db)
        silence_ratio = num_silent / len(y)
        
        # 3. Dynamic Range
        rms = np.sqrt(np.mean(y**2))
        peak = np.max(np.abs(y))
        dynamic_range = 20 * np.log10(peak / rms) if rms > 0 else 0.0
        
        # 4. Musicality (New)
        beat_stability = compute_beat_stability(y, sr)
        key_consistency = compute_key_consistency(y, sr)
            
        return {
            "clipping_ratio": float(clipping_ratio),
            "silence_ratio": float(silence_ratio),
            "dynamic_range_db": float(dynamic_range),
            "duration_sec": float(len(y) / sr),
            "beat_stability": beat_stability,
            "key_consistency": key_consistency
        }
    except Exception as e:
        print(f"Error computing quality metrics for {audio_path}: {e}")
        return {}

class CLAPScorer:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        if not HAS_CLAP:
            print("Warning: laion_clap not installed. CLAP scores will be skipped.")
            self.model = None
            return
        print(f"Loading CLAP model on {device}...")
        # Use HTSAT-tiny which matches the default checkpoint
        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
        self.model.load_ckpt()  # downloads default ckpt
        self.device = device
        
    def score(self, audio_paths: List[str], texts: List[str]) -> List[float]:
        if not HAS_CLAP: return [0.0] * len(audio_paths)
        try:
            audio_embed = self.model.get_audio_embedding_from_filelist(x=audio_paths, use_tensor=True)
            text_embed = self.model.get_text_embedding(texts, use_tensor=True)
            
            audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1)
            text_embed = torch.nn.functional.normalize(text_embed, dim=-1)
            
            similarity = (audio_embed * text_embed).sum(dim=1)
            return similarity.cpu().tolist()
        except Exception as e:
            print(f"Error calculating CLAP score: {e}")
            return [0.0] * len(audio_paths)

class FADScorer:
    def __init__(self, reference_dir: str, model_name="vggish", sample_rate=16000):
        if not HAS_FAD:
            print("Warning: frechet_audio_distance not installed. FAD will be skipped.")
            return
        if not os.path.exists(reference_dir):
            print(f"Warning: Reference directory {reference_dir} does not exist. FAD skipped.")
            self.frechet = None
            return
        print(f"Initializing FAD with model {model_name} and background {reference_dir}...")
        self.frechet = FrechetAudioDistance(
            model_name=model_name,
            sample_rate=sample_rate,
            use_pca=False, use_activation=False, verbose=False
        )
        self.reference_dir = reference_dir
        
    def score(self, generated_dir: str) -> float:
        if not HAS_FAD or self.frechet is None: return 0.0
        try:
            fad_score = self.frechet.score(self.reference_dir, generated_dir, dtype="float32")
            return float(fad_score)
        except Exception as e:
            print(f"Error calculating FAD: {e}")
            return 0.0


# --- Generation Logic ---

def convert_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "192k"):
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-b:a", bitrate, mp3_path],
            check=True, capture_output=True
        )
        os.remove(wav_path)
        return True
    except: return False

def generate_eval_set(model, output_dir, device):
    """Generates audio for all EVAL_PROMPTS and EVAL_SEEDS."""
    os.makedirs(output_dir, exist_ok=True)
    prompt_map = {}
    generated_files = []

    print(f"Generating {len(EVAL_PROMPTS) * len(EVAL_SEEDS)} clips...")
    
    for prompt_name, prompt_text in tqdm(EVAL_PROMPTS.items(), desc="Prompts"):
        for seed in EVAL_SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
                
            filename = f"{prompt_name}_s{seed}.wav"
            filepath = os.path.join(output_dir, filename)
            
            if os.path.exists(filepath):
                generated_files.append(filepath)
                prompt_map[filename] = prompt_text
                continue
                
            try:
                # Temporarily override max length for speed using object.__setattr__ to bypass Pydantic frozen check
                original_length = model.config.data.audio_length
                object.__setattr__(model.config.data, 'audio_length', 512)
                
                with torch.inference_mode(), torch.amp.autocast('cuda', enabled=False):
                    output_audio = model.generate(
                        text=prompt_text,
                        cfg_scale=DEFAULT_PARAMS["cfg_scale"],
                        temperature=DEFAULT_PARAMS["temperature"],
                        top_p=DEFAULT_PARAMS["top_p"],
                    )
                
                # Restore config
                object.__setattr__(model.config.data, 'audio_length', original_length)
                
                if isinstance(output_audio, np.ndarray):
                    if output_audio.ndim == 2 and output_audio.shape[0] < output_audio.shape[1]:
                        output_audio = output_audio.T
                
                sf.write(filepath, output_audio, 44100)
                generated_files.append(filepath)
                prompt_map[filename] = prompt_text
            except Exception as e:
                print(f"Error generating {filename}: {e}")

    with open(os.path.join(output_dir, "prompts.json"), 'w') as f:
        json.dump(prompt_map, f, indent=2)
    return generated_files, prompt_map

def evaluate_checkpoint(ckpt_path, config_path, eval_root, device="cuda", skip_gen=False, reference_dir=None):
    """Full pipeline for a single checkpoint."""
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    output_dir = os.path.join(eval_root, "audio", ckpt_name)
    
    print(f"\n=== Evaluating {ckpt_name} ===")
    
    if not skip_gen:
        print(f"Loading model from {ckpt_path}...")
        try:
            model = Dia.from_local(config_path, ckpt_path, device=torch.device(device))
            model.model = model.model.float()
            model.model.eval()
            generated_files, prompt_map = generate_eval_set(model, output_dir, device)
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Failed to load/run model: {e}")
            return None
    else:
        print("Skipping generation...")
        generated_files = glob.glob(os.path.join(output_dir, "*.wav"))
        try:
            with open(os.path.join(output_dir, "prompts.json"), 'r') as f:
                prompt_map = json.load(f)
        except: prompt_map = {}

    print("Calculating metrics...")
    metrics_results = {
        "checkpoint": ckpt_name,
        "step": int(ckpt_name.split('step')[-1]) if 'step' in ckpt_name else 0
    }
    
    # Basic Quality
    if generated_files:
        qual_scores = [compute_audio_quality_metrics(f) for f in generated_files]
        df_qual = pd.DataFrame(qual_scores)
        metrics_results["clipping_ratio"] = float(df_qual["clipping_ratio"].mean()) if not df_qual.empty else 0.0
        metrics_results["silence_ratio"] = float(df_qual["silence_ratio"].mean()) if not df_qual.empty else 0.0
        metrics_results["beat_stability"] = float(df_qual["beat_stability"].mean()) if not df_qual.empty else 1.0
        metrics_results["key_consistency"] = float(df_qual["key_consistency"].mean()) if not df_qual.empty else 0.0
    
    # CLAP
    try:
        scorer = CLAPScorer(device=device)
        files, texts = [], []
        for fname, text in prompt_map.items():
            fpath = os.path.join(output_dir, fname)
            if os.path.exists(fpath):
                files.append(fpath)
                texts.append(text)
        if files:
            scores = scorer.score(files, texts)
            metrics_results["clap_score"] = float(np.mean(scores))
    except Exception as e:
        print(f"CLAP failed: {e}")
        metrics_results["clap_score"] = 0.0
    
    # FAD
    if reference_dir and HAS_FAD:
        try:
            fad_scorer = FADScorer(reference_dir)
            metrics_results["fad_score"] = fad_scorer.score(output_dir)
        except Exception as e:
            print(f"FAD failed: {e}")
            metrics_results["fad_score"] = 0.0

    return metrics_results

def main():
    parser = argparse.ArgumentParser()
    # Mode switch
    subparsers = parser.add_subparsers(dest="mode", help="Mode: 'checkpoint-eval' or 'folder-eval'")
    
    # Mode 1: Checkpoint Evaluation (Generation + Metrics)
    parser_ckpt = subparsers.add_parser("checkpoint-eval", help="Evaluate multiple checkpoints")
    parser_ckpt.add_argument("--checkpoint_dir", type=str, required=True, help="Folder with .pt checkpoints")
    parser_ckpt.add_argument("--config", type=str, default="./configs/architecture/experiments/20251124_dia_003_model_inference.json")
    parser_ckpt.add_argument("--output_dir", type=str, default="evaluation_results")
    parser_ckpt.add_argument("--reference_dir", type=str, help="Path to real audio for FAD")
    parser_ckpt.add_argument("--every_n", type=int, default=1, help="Process every Nth checkpoint")
    parser_ckpt.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser_ckpt.add_argument("--skip_gen", action="store_true", help="Skip generation, only run metrics")

    # Mode 2: Folder Evaluation (Metrics Only)
    parser_folder = subparsers.add_parser("folder-eval", help="Evaluate an existing folder of audio")
    parser_folder.add_argument("--folder", type=str, required=True, help="Folder with audio files")
    parser_folder.add_argument("--prompts_file", type=str, help="JSON file for CLAP prompts")
    parser_folder.add_argument("--reference_dir", type=str, help="Path to real audio for FAD")
    parser_folder.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    if args.mode == "checkpoint-eval":
        # Find checkpoints (both .pt and .pth)
        checkpoints = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.pt")) + 
                             glob.glob(os.path.join(args.checkpoint_dir, "*.pth")))
        
        if args.every_n > 1: checkpoints = checkpoints[::args.every_n]
        
        if not checkpoints:
            print(f"No checkpoints found in {args.checkpoint_dir}")
            return

        print(f"Found {len(checkpoints)} checkpoints to evaluate.")
        all_results = []
        for ckpt in checkpoints:
            res = evaluate_checkpoint(ckpt, args.config, args.output_dir, args.device, args.skip_gen, args.reference_dir)
            if res: all_results.append(res)
            
        df = pd.DataFrame(all_results)
        report_path = os.path.join(args.output_dir, "final_report.csv")
        df.to_csv(report_path, index=False)
        print(f"\nEvaluation Complete! Report saved to {report_path}")
        print(df)

    elif args.mode == "folder-eval":
        print(f"Evaluating folder: {args.folder}")
        files = glob.glob(os.path.join(args.folder, "*.wav")) + glob.glob(os.path.join(args.folder, "*.mp3"))
        
        # Quality
        qual_scores = [compute_audio_quality_metrics(f) for f in tqdm(files)]
        df_qual = pd.DataFrame(qual_scores)
        print("\nQuality Metrics:")
        print(df_qual.mean(numeric_only=True))
        
        # CLAP
        if args.prompts_file:
            scorer = CLAPScorer(device=args.device)
            with open(args.prompts_file, 'r') as f: prompt_map = json.load(f)
            valid_files, valid_texts = [], []
            for f in files:
                name = os.path.basename(f)
                if name in prompt_map:
                    valid_files.append(f)
                    valid_texts.append(prompt_map[name])
            if valid_files:
                scores = scorer.score(valid_files, valid_texts)
                print(f"Average CLAP Score: {np.mean(scores):.4f}")
        
        # FAD
        if args.reference_dir:
            fad_scorer = FADScorer(args.reference_dir)
            print(f"FAD Score: {fad_scorer.score(args.folder)}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
