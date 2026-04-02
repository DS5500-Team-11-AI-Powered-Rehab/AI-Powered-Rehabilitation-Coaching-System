"""
Integration Layer Testing Script
Tests integration layer with real CV output from QEVD dataset

Usage:
    python test_integration.py --dataset test
    python test_integration.py --dataset train
    python test_integration.py --dataset val
    python test_integration.py --dataset both
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import argparse

# Add src to path (use relative path from this file)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from integration.integration_layer import IntegrationLayer, Config


# ==========================================
# CONFIGURATION
# ==========================================

CV_OUTPUT_BASE = Path(__file__).resolve().parent

DATASETS = {
    'test': CV_OUTPUT_BASE / "rag_infer_logs_test",
    'train': CV_OUTPUT_BASE / "rag_infer_logs_train",
    'val': CV_OUTPUT_BASE / "rag_infer_logs_val",
    'sample': CV_OUTPUT_BASE / "sample_cv_output",  # Small sample data for quick testing
    'synthetic': CV_OUTPUT_BASE / "synthetic_test_data",  # Generated synthetic data from test
    'synthetic_train': CV_OUTPUT_BASE / "synthetic_train_data",  # Generated synthetic data from train
    'synthetic_val': CV_OUTPUT_BASE / "synthetic_val_data",  # Generated synthetic data from val
}

RESULTS_DIR = Path("./test_results")
RESULTS_DIR.mkdir(exist_ok=True)


# ==========================================
# CV OUTPUT LOADER
# ==========================================

def load_cv_outputs(dataset_dir: Path):
    """
    Load all CV output JSON files from dataset directory
    Supports both .json and .jsonl.gz formats
    
    .jsonl.gz format (QEVD dataset):
        - Line 1: {"__meta__": {...}}
        - Line 2+: Individual frame predictions
    
    .json format (sample data):
        - Array of frame predictions
    
    Returns:
        List of (video_file, frames) tuples
    """
    import gzip
    
    videos = []
    
    # Try loading .jsonl.gz files first (QEVD dataset format)
    gz_files = list(dataset_dir.glob("*.jsonl.gz"))
    
    if gz_files:
        print(f"ℹ️  Found {len(gz_files)} .jsonl.gz files, loading...")
        for gz_file in sorted(gz_files):
            try:
                with gzip.open(gz_file, 'rt') as f:
                    lines = f.readlines()
                    
                    if len(lines) <= 1:
                        # Only metadata, no frame data
                        continue
                    
                    # Skip first line (metadata), parse remaining lines as frames
                    frames = []
                    for line in lines[1:]:
                        line = line.strip()
                        if line:  # Skip empty lines
                            try:
                                frame = json.loads(line)
                                frames.append(frame)
                            except json.JSONDecodeError:
                                continue
                    
                    if len(frames) > 0:
                        videos.append({
                            'filename': gz_file.name,
                            'video_path': gz_file.stem.replace('.jsonl', ''),
                            'frames': frames,
                            'num_frames': len(frames)
                        })
                        
            except Exception as e:
                print(f"⚠️  Error loading {gz_file.name}: {e}")
                continue
    
    # Fallback to .json files (sample data format)
    else:
        json_files = list(dataset_dir.glob("*.json"))
        # Filter out manifest files
        json_files = [f for f in json_files if f.name != "manifest.json"]
        
        print(f"ℹ️  Found {len(json_files)} .json files, loading...")
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Andre's output format: list of frame predictions
                if isinstance(data, list) and len(data) > 0:
                    frames = data
                # Or single prediction
                elif isinstance(data, dict) and 'predictions' in data:
                    frames = data['predictions']
                else:
                    # Skip metadata-only files
                    continue
                
                if len(frames) > 0:  # Only add videos with frames
                    videos.append({
                        'filename': json_file.name,
                        'video_path': json_file.stem,
                        'frames': frames,
                        'num_frames': len(frames)
                    })
                
            except Exception as e:
                print(f"⚠️  Error loading {json_file.name}: {e}")
                continue
    
    return videos


# ==========================================
# INTEGRATION LAYER TESTING
# ==========================================

class IntegrationTester:
    """
    Test integration layer with CV outputs
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.results = {
            'total_videos': 0,
            'total_frames': 0,
            'total_coaching_events': 0,
            'videos_with_events': 0,
            'tier_breakdown': {'tier_1': 0, 'tier_2': 0, 'tier_3': 0},
            'severity_breakdown': {'high': 0, 'medium': 0, 'low': 0},
            'mistake_types': defaultdict(int),
            'exercise_types': defaultdict(int),
            'events_per_video': [],
            'video_details': []
        }
    
    def test_video(self, video_data: dict, verbose: bool = False):
        """
        Test integration layer on a single video
        """
        
        video_name = video_data['filename']
        frames = video_data['frames']
        
        # Create new integration layer for this video (new session)
        session_id = f"test_{video_name.replace('.json', '')}"
        integration_layer = IntegrationLayer(session_id=session_id, config=self.config)
        
        # Populate cache (needed for Tier 1 routing)
        integration_layer.cache.populate_defaults()
        
        # Track events for this video
        video_events = []
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing: {video_name}")
            print(f"Frames: {len(frames)}")
            print(f"{'='*60}")
        
        # Process each frame
        for i, frame in enumerate(frames):
            
            # Process through integration layer
            coaching_event = integration_layer.process_frame(frame)
            
            if coaching_event:
                video_events.append(coaching_event)
                
                if verbose:
                    print(f"\n[Frame {i}] 🎯 COACHING EVENT")
                    print(f"  Exercise: {coaching_event['exercise']['name']}")
                    print(f"  Mistake: {coaching_event['mistake']['type']}")
                    print(f"  Severity: {coaching_event['severity']}")
                    print(f"  Duration: {coaching_event['mistake']['duration_seconds']:.1f}s")
                    print(f"  Persistence: {coaching_event['mistake']['persistence_rate']:.1%}")
                    print(f"  Tier: {coaching_event['tier']}")
                    print(f"  Reason: {coaching_event['routing_reason']}")
        
        # Update results
        self.results['total_videos'] += 1
        self.results['total_frames'] += len(frames)
        self.results['total_coaching_events'] += len(video_events)
        
        if video_events:
            self.results['videos_with_events'] += 1
        
        self.results['events_per_video'].append(len(video_events))
        
        # Track event details
        for event in video_events:
            self.results['tier_breakdown'][event['tier']] += 1
            self.results['severity_breakdown'][event['severity']] += 1
            self.results['mistake_types'][event['mistake']['type']] += 1
            self.results['exercise_types'][event['exercise']['name']] += 1
        
        # Store video details
        video_detail = {
            'filename': video_name,
            'num_frames': len(frames),
            'num_events': len(video_events),
            'events': [
                {
                    'mistake': e['mistake']['type'],
                    'exercise': e['exercise']['name'],
                    'severity': e['severity'],
                    'tier': e['tier'],
                    'duration': e['mistake']['duration_seconds'],
                    'persistence': e['mistake']['persistence_rate']
                }
                for e in video_events
            ]
        }
        self.results['video_details'].append(video_detail)
        
        if verbose and video_events:
            print(f"\n✅ {len(video_events)} coaching events generated for {video_name}")
        elif verbose:
            print(f"\n⚪ No coaching events for {video_name}")
        
        return video_events
    
    def test_dataset(self, dataset_name: str, verbose: bool = False, max_videos: int = None):
        """
        Test integration layer on entire dataset
        """
        
        print(f"\n{'='*60}")
        print(f"TESTING DATASET: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        dataset_dir = DATASETS[dataset_name]
        
        if not dataset_dir.exists():
            print(f"❌ Dataset directory not found: {dataset_dir}")
            return
        
        print(f"Loading CV outputs from: {dataset_dir}")
        videos = load_cv_outputs(dataset_dir)
        
        if not videos:
            print(f"❌ No CV output files found in {dataset_dir}")
            return
        
        print(f"✅ Loaded {len(videos)} videos")

        # Detect event-log style datasets (single-frame records) that cannot
        # satisfy temporal persistence thresholds used by the integration layer.
        avg_frames = sum(v['num_frames'] for v in videos) / len(videos)
        if avg_frames < self.config.MIN_FRAMES:
            print(
                "⚠️  Dataset appears to contain short event logs, not continuous frame sequences."
            )
            print(
                f"   Avg frames/video: {avg_frames:.2f} (MIN_FRAMES={self.config.MIN_FRAMES})"
            )
            print(
                "   This will typically produce 0 coaching events because persistence/duration checks cannot pass."
            )
            print(
                "   Suggested next step: generate synthetic continuous frames with generate_synthetic_data.py"
            )
        
        # Limit if specified
        if max_videos:
            videos = videos[:max_videos]
            print(f"Testing first {max_videos} videos only")
        
        print(f"\nProcessing videos...")
        
        # Process each video
        for i, video in enumerate(videos):
            if not verbose and i % 10 == 0:
                print(f"  Processed {i}/{len(videos)} videos...", end='\r')
            
            self.test_video(video, verbose=verbose)
        
        if not verbose:
            print(f"  Processed {len(videos)}/{len(videos)} videos... ✅")
        
        print(f"\n{'='*60}")
        print(f"TESTING COMPLETE")
        print(f"{'='*60}")
    
    def print_summary(self):
        """
        Print test results summary
        """
        
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        
        # Overall stats
        print(f"\n📊 Overall Statistics:")
        print(f"  Total Videos: {self.results['total_videos']}")
        print(f"  Total Frames: {self.results['total_frames']}")
        print(f"  Total Coaching Events: {self.results['total_coaching_events']}")
        
        if self.results['total_videos'] > 0:
            coverage_pct = self.results['videos_with_events']/self.results['total_videos']*100
            print(f"  Videos with Events: {self.results['videos_with_events']} ({coverage_pct:.1f}%)")
            avg_events = self.results['total_coaching_events'] / self.results['total_videos']
            print(f"  Avg Events per Video: {avg_events:.2f}")
        else:
            print(f"  Videos with Events: 0 (0%)")
            print(f"  Avg Events per Video: 0")
        
        # Tier breakdown
        print(f"\n🎯 Tier Breakdown:")
        total_events = self.results['total_coaching_events']
        if total_events > 0:
            for tier in ['tier_1', 'tier_2', 'tier_3']:
                count = self.results['tier_breakdown'][tier]
                pct = count / total_events * 100
                print(f"  {tier.replace('_', ' ').title()}: {count} ({pct:.1f}%)")
            
            # Cache hit rate
            cache_hit_rate = self.results['tier_breakdown']['tier_1'] / total_events * 100
            print(f"\n  💾 Cache Hit Rate: {cache_hit_rate:.1f}%")
        
        # Severity breakdown
        print(f"\n⚠️  Severity Breakdown:")
        if total_events > 0:
            for severity in ['high', 'medium', 'low']:
                count = self.results['severity_breakdown'][severity]
                pct = count / total_events * 100
                print(f"  {severity.title()}: {count} ({pct:.1f}%)")
        
        # Top mistakes
        print(f"\n🎯 Top 10 Mistake Types:")
        sorted_mistakes = sorted(self.results['mistake_types'].items(), key=lambda x: x[1], reverse=True)[:10]
        for mistake, count in sorted_mistakes:
            print(f"  {mistake}: {count}")
        
        # Top exercises
        print(f"\n🏋️  Top 10 Exercise Types:")
        sorted_exercises = sorted(self.results['exercise_types'].items(), key=lambda x: x[1], reverse=True)[:10]
        for exercise, count in sorted_exercises:
            print(f"  {exercise}: {count}")
        
        # Videos with most events
        print(f"\n📹 Top 5 Videos with Most Events:")
        sorted_videos = sorted(self.results['video_details'], key=lambda x: x['num_events'], reverse=True)[:5]
        for video in sorted_videos:
            print(f"  {video['filename']}: {video['num_events']} events")
        
        # Videos with no events
        videos_no_events = [v for v in self.results['video_details'] if v['num_events'] == 0]
        print(f"\n⚪ Videos with No Events: {len(videos_no_events)}")
        if len(videos_no_events) <= 10:
            for video in videos_no_events:
                print(f"  {video['filename']}")
        else:
            print(f"  (Too many to list - {len(videos_no_events)} total)")
    
    def save_results(self, dataset_name: str):
        """
        Save detailed results to JSON
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = RESULTS_DIR / f"test_results_{dataset_name}_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        return output_file


# ==========================================
# MAIN TESTING FUNCTION
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Test Integration Layer with CV Outputs')
    parser.add_argument('--dataset', choices=['test', 'train', 'val', 'sample', 'both', 'synthetic', 'synthetic_train', 'synthetic_val'], default='train',
                       help='Which dataset to test on (test/train/val=CV event logs, sample=small test, synthetic=generated data)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed output for each video')
    parser.add_argument('--max-videos', type=int, default=None,
                       help='Limit number of videos to test (for quick testing)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom config JSON')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = Config()
        for key, value in config_dict.items():
            setattr(config, key, value)
    else:
        config = Config()
    
    print("\n" + "="*60)
    print("INTEGRATION LAYER TESTING")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  MIN_FRAMES: {config.MIN_FRAMES}")
    print(f"  MIN_PERSISTENCE_RATE: {config.MIN_PERSISTENCE_RATE}")
    print(f"  MIN_CONFIDENCE: {config.MIN_CONFIDENCE}")
    print(f"  MIN_DURATION_SECONDS: {config.MIN_DURATION_SECONDS}")
    print(f"  MIN_COACHING_INTERVAL: {config.MIN_COACHING_INTERVAL}")
    
    # Test datasets
    datasets_to_test = ['test', 'val'] if args.dataset == 'both' else [args.dataset]
    
    for dataset_name in datasets_to_test:
        tester = IntegrationTester(config=config)
        tester.test_dataset(dataset_name, verbose=args.verbose, max_videos=args.max_videos)
        tester.print_summary()
        tester.save_results(dataset_name)


if __name__ == "__main__":
    main()


# ==========================================
# QUICK TEST EXAMPLES
# ==========================================

"""
USAGE EXAMPLES:

# 1. Quick test on first 10 videos
python test_integration.py --dataset train --max-videos 10

# 2. Full test on validation set
python test_integration.py --dataset val

# 3. Test both datasets
python test_integration.py --dataset both

# 4. Verbose output (see each event)
python test_integration.py --dataset train --max-videos 5 --verbose

# 5. Test with custom config
python test_integration.py --dataset train --config custom_config.json
"""
