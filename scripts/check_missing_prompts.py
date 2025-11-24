import os
import argparse
from collections import defaultdict
from pathlib import Path

def get_song_id(filename):
    """
    Extracts the song ID from a filename.
    Assumes the format: 'Song Name [VideoID]_4bars_XXX...'
    Returns the part before '_4bars'.
    """
    if "_4bars" in filename:
        return filename.split("_4bars")[0]
    return None

def main():
    parser = argparse.ArgumentParser(description="Check for missing audio prompts.")
    parser.add_argument("--audio-dir", required=True, help="Directory containing .wav audio files")
    parser.add_argument("--prompts-dir", required=True, help="Directory containing .txt prompt files")
    
    args = parser.parse_args()
    
    audio_dir = Path(args.audio_dir)
    prompts_dir = Path(args.prompts_dir)
    
    if not audio_dir.exists():
        print(f"Error: Audio directory not found: {audio_dir}")
        return
    if not prompts_dir.exists():
        print(f"Error: Prompts directory not found: {prompts_dir}")
        return

    print(f"Scanning audio directory: {audio_dir}")
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    print(f"Found {len(audio_files)} audio files.")

    print(f"Scanning prompts directory: {prompts_dir}")
    prompt_files = set(os.listdir(prompts_dir))
    print(f"Found {len(prompt_files)} prompt files.")

    # Dictionary to track missing segments per song
    # Key: Song ID, Value: List of missing segment filenames
    missing_per_song = defaultdict(list)
    
    # Dictionary to track total segments per song to distinguish partial vs full missing
    total_segments_per_song = defaultdict(int)
    
    # Set of all unique song IDs found in audio
    all_songs = set()
    
    total_missing_segments = 0

    for audio_file in audio_files:
        song_id = get_song_id(audio_file)
        if song_id:
            all_songs.add(song_id)
            total_segments_per_song[song_id] += 1

        # Construct expected prompt filename
        # Audio:  Song..._4bars_001.wav
        # Prompt: Song..._4bars_001_prompt.txt
        
        base_name = os.path.splitext(audio_file)[0]
        expected_prompt = f"{base_name}_prompt.txt"
        
        if expected_prompt not in prompt_files:
            if song_id:
                missing_per_song[song_id].append(audio_file)
            total_missing_segments += 1

    # Analysis
    songs_with_missing = list(missing_per_song.keys())
    fully_prompted_songs = len(all_songs) - len(songs_with_missing)
    
    completely_unprompted_songs = 0
    partially_prompted_songs = 0
    
    for song_id in songs_with_missing:
        missing_count = len(missing_per_song[song_id])
        total_count = total_segments_per_song[song_id]
        
        if missing_count == total_count:
            completely_unprompted_songs += 1
        else:
            partially_prompted_songs += 1

    # Print details of missing songs first (so summary is at the end)
    if len(songs_with_missing) > 0:
        print("\n" + "="*50)
        print("SONGS WITH MISSING PROMPTS")
        print("="*50)
        # Sort by number of missing segments (descending)
        sorted_missing = sorted(missing_per_song.items(), key=lambda x: len(x[1]), reverse=True)
        
        for song_id, missing_files in sorted_missing:
            status = "COMPLETELY MISSING" if len(missing_files) == total_segments_per_song[song_id] else "PARTIALLY MISSING"
            print(f"\nSong: {song_id}")
            print(f"  Status: {status}")
            print(f"  Missing {len(missing_files)} / {total_segments_per_song[song_id]} segments")

    # Print Summary
    print("\n" + "="*50)
    print("ANALYSIS RESULTS")
    print("="*50)
    print(f"Total unique songs found: {len(all_songs)}")
    print(f"Songs fully prompted: {fully_prompted_songs}")
    print(f"Songs partially prompted: {partially_prompted_songs}")
    print(f"Songs completely unprompted: {completely_unprompted_songs}")
    print("-" * 30)
    print(f"Total songs with missing segments: {len(songs_with_missing)}")
    print(f"Total missing prompt files: {total_missing_segments}")
    print("="*50)

if __name__ == "__main__":
    main()
