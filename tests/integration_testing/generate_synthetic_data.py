#!/usr/bin/env python3
"""
Generate Synthetic Continuous Frame Data from Event Logs

Takes single-frame event predictions from .jsonl.gz files and generates
synthetic continuous frame sequences for testing the integration layer.

Usage:
    python generate_synthetic_data.py --input rag_infer_logs_test --output synthetic_test_data --num-videos 20
"""

import argparse
import gzip
import json
from pathlib import Path
from typing import List, Dict
import random
import copy


def load_event_frame(gz_file: Path) -> tuple:
    """
    Load a single event frame from a .jsonl.gz file
    Returns: (metadata, frame) or (metadata, None) if no frame
    """
    with gzip.open(gz_file, 'rt') as f:
        lines = f.readlines()
    
    metadata = json.loads(lines[0])
    frame = json.loads(lines[1]) if len(lines) > 1 else None
    
    return metadata, frame


def generate_frame_sequence(event_frame: Dict, metadata: Dict, sequence_length: int = 100) -> List[Dict]:
    """
    Generate a synthetic frame sequence around an event frame
    
    Strategy:
    - Start with "good form" frames (no mistakes)
    - Gradually introduce the mistake(s) from the event frame
    - Keep mistake persistent for a while
    - Optionally resolve it at the end
    
    Args:
        event_frame: The frame with detected mistakes/events
        metadata: Video metadata
        sequence_length: Total frames to generate
        
    Returns:
        List of synthetic frames
    """
    
    frames = []
    fps = metadata['__meta__'].get('fps', 30.0)
    
    # Extract event frame info
    exercise = event_frame.get('exercise', {'name': 'unknown', 'p': 0.8})
    mistakes = event_frame.get('mistakes', [])
    metrics = event_frame.get('metrics', {})
    
    # Determine mistake appearance pattern
    # We want the mistake to appear and persist for testing
    mistake_start_frame = random.randint(15, 30)  # Mistake starts
    mistake_end_frame = random.randint(70, sequence_length)  # Mistake may resolve
    
    for i in range(sequence_length):
        timestamp_s = i / fps
        
        # Determine if mistakes should be present in this frame
        if mistake_start_frame <= i < mistake_end_frame:
            # Mistake is present - use with varying confidence
            # Add some variation so it's realistic
            frame_mistakes = []
            for mistake in mistakes:
                # Vary confidence slightly frame-to-frame
                confidence_variation = random.uniform(-0.1, 0.1)
                varied_confidence = max(0.3, min(0.95, mistake['p'] + confidence_variation))
                
                # Sometimes the mistake is detected, sometimes not (realistic)
                if random.random() < 0.7:  # 70% detection rate while mistake persists
                    frame_mistakes.append({
                        'name': mistake['name'],
                        'p': varied_confidence
                    })
        else:
            # No mistakes in this frame, or use different mistakes
            frame_mistakes = []
            
            # Occasionally add minor transient mistakes (not persistent)
            if random.random() < 0.1:  # 10% chance of transient issue
                transient_mistakes = [
                    {'name': 'minor wobble', 'p': random.uniform(0.3, 0.5)},
                    {'name': 'slight shift', 'p': random.uniform(0.3, 0.5)}
                ]
                frame_mistakes.append(random.choice(transient_mistakes))
        
        # Create synthetic frame
        frame = {
            'timestamp_s': timestamp_s,
            'frame_index': i,
            'source_fps': fps,
            'exercise': {
                'name': exercise['name'],
                'p': exercise['p'] + random.uniform(-0.05, 0.05)
            },
            'mistakes': frame_mistakes,
            'metrics': {
                'speed_rps': metrics.get('speed_rps', 1.0) + random.uniform(-0.1, 0.1),
                'rom_level': metrics.get('rom_level', 2),
                'height_level': metrics.get('height_level', 3),
                'torso_rotation': metrics.get('torso_rotation', 0),
                'direction': metrics.get('direction', 'none'),
                'no_obvious_issue_p': 0.9 if not frame_mistakes else random.uniform(0.1, 0.3)
            },
            'quality_score': random.uniform(0.6, 0.9) if not frame_mistakes else random.uniform(0.2, 0.5),
            'speak_now': 0.0
        }
        
        frames.append(frame)
    
    return frames


def generate_synthetic_dataset(input_dir: Path, output_dir: Path, num_videos: int = 20, frames_per_video: int = 100):
    """
    Generate synthetic continuous frame data from event logs
    
    Args:
        input_dir: Directory with .jsonl.gz event files
        output_dir: Where to save synthetic .json files
        num_videos: How many synthetic videos to create
        frames_per_video: Frames per synthetic video
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating synthetic dataset:")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Videos: {num_videos}")
    print(f"  Frames per video: {frames_per_video}")
    print()
    
    # Find all .jsonl.gz files with events
    gz_files = [f for f in input_dir.glob("*.jsonl.gz")]
    
    # Filter to only files with frame data
    event_files = []
    for gz_file in gz_files:
        metadata, frame = load_event_frame(gz_file)
        if frame is not None:
            event_files.append((gz_file, metadata, frame))
    
    print(f"Found {len(event_files)} files with event frames")
    
    if len(event_files) == 0:
        print("❌ No event frames found!")
        return
    
    # Generate synthetic videos
    num_to_generate = min(num_videos, len(event_files))
    selected_files = random.sample(event_files, num_to_generate)
    
    print(f"Generating {num_to_generate} synthetic videos...\n")
    
    for idx, (gz_file, metadata, event_frame) in enumerate(selected_files, 1):
        # Generate frame sequence
        frames = generate_frame_sequence(event_frame, metadata, frames_per_video)
        
        # Save to JSON file
        video_name = metadata['__meta__']['name']
        output_file = output_dir / f"{video_name}_synthetic.json"
        
        with open(output_file, 'w') as f:
            json.dump(frames, f, indent=2)
        
        # Print summary
        mistake_types = {}
        for frame in frames:
            for mistake in frame.get('mistakes', []):
                name = mistake['name']
                mistake_types[name] = mistake_types.get(name, 0) + 1
        
        print(f"✅ {idx}/{num_to_generate}: {output_file.name}")
        print(f"   Exercise: {event_frame['exercise']['name']}")
        print(f"   Frames: {len(frames)}")
        print(f"   Mistakes detected:")
        for mistake_name, count in mistake_types.items():
            persistence = count / len(frames) * 100
            print(f"     - {mistake_name}: {count}/{len(frames)} frames ({persistence:.1f}%)")
        print()
    
    print(f"\n🎉 Successfully generated {num_to_generate} synthetic videos")
    print(f"   Saved to: {output_dir}")
    print(f"\nTest with:")
    print(f"   python test_integration.py --dataset synthetic --max-videos 10 --verbose")


def main():
    parser = argparse.ArgumentParser(description='Generate Synthetic CV Output Data')
    parser.add_argument('--input', type=str, default='rag_infer_logs_test',
                       help='Input directory with .jsonl.gz event files')
    parser.add_argument('--output', type=str, default='synthetic_test_data',
                       help='Output directory for synthetic .json files')
    parser.add_argument('--num-videos', type=int, default=20,
                       help='Number of synthetic videos to generate')
    parser.add_argument('--frames-per-video', type=int, default=100,
                       help='Frames per synthetic video')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    generate_synthetic_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        num_videos=args.num_videos,
        frames_per_video=args.frames_per_video
    )


if __name__ == "__main__":
    main()


# ==========================================
# USAGE EXAMPLES
# ==========================================

"""
# Generate 20 synthetic videos from test data
python generate_synthetic_data.py --input rag_infer_logs_test --output synthetic_test_data --num-videos 20

# Generate 50 synthetic videos from validation data with more frames
python generate_synthetic_data.py --input rag_infer_logs_val --output synthetic_val_data --num-videos 50 --frames-per-video 150

# Generate both
python generate_synthetic_data.py --input rag_infer_logs_test --output synthetic_test_data --num-videos 30
python generate_synthetic_data.py --input rag_infer_logs_val --output synthetic_val_data --num-videos 30
"""
