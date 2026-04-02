"""
Threshold Tuning Script
Helps you optimize IntegrationLayer configuration based on test results

Usage:
    python test_tune_thresholds.py --results test_results_test_20240305_143022.json
"""

import json
import argparse
from pathlib import Path
import sys

# Add src to path (use relative path from this file)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from integration.integration_layer import Config


# ==========================================
# THRESHOLD TUNING ANALYZER
# ==========================================

class ThresholdTuner:
    """
    Analyzes test results and suggests threshold adjustments
    """
    
    def __init__(self, results_file: Path):
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        self.current_config = Config()
    
    def analyze(self):
        """
        Analyze results and provide tuning recommendations
        """
        
        print("\n" + "="*60)
        print("THRESHOLD TUNING ANALYSIS")
        print("="*60)
        
        total_videos = self.results['total_videos']
        total_events = self.results['total_coaching_events']
        videos_with_events = self.results['videos_with_events']
        
        # Calculate key metrics
        event_rate = total_events / total_videos if total_videos > 0 else 0
        video_coverage = videos_with_events / total_videos * 100 if total_videos > 0 else 0
        
        cache_hit_rate = 0
        if total_events > 0:
            cache_hit_rate = self.results['tier_breakdown']['tier_1'] / total_events * 100
        
        print(f"\nCurrent Configuration:")
        print(f"  MIN_PERSISTENCE_RATE: {self.current_config.MIN_PERSISTENCE_RATE}")
        print(f"  MIN_CONFIDENCE: {self.current_config.MIN_CONFIDENCE}")
        print(f"  MIN_DURATION_SECONDS: {self.current_config.MIN_DURATION_SECONDS}")
        print(f"  MIN_COACHING_INTERVAL: {self.current_config.MIN_COACHING_INTERVAL}")
        
        print(f"\nCurrent Metrics:")
        print(f"  Events per Video: {event_rate:.2f}")
        print(f"  Video Coverage: {video_coverage:.1f}%")
        print(f"  Cache Hit Rate: {cache_hit_rate:.1f}%")
        
        # Provide recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        recommendations = []
        
        # Check if too many events
        if event_rate > 5:
            print("\n⚠️  TOO MANY COACHING EVENTS")
            print(f"  Current: {event_rate:.2f} events per video")
            print(f"  Target: 1-3 events per video")
            print("\n  Recommendations:")
            print(f"    1. Increase MIN_PERSISTENCE_RATE from {self.current_config.MIN_PERSISTENCE_RATE} to 0.35-0.40")
            print(f"    2. Increase MIN_DURATION_SECONDS from {self.current_config.MIN_DURATION_SECONDS} to 4.0-5.0")
            print(f"    3. Increase MIN_CONFIDENCE from {self.current_config.MIN_CONFIDENCE} to 0.40")
            
            recommendations.append({
                'MIN_PERSISTENCE_RATE': 0.35,
                'MIN_DURATION_SECONDS': 4.0,
                'MIN_CONFIDENCE': 0.40
            })
        
        # Check if too few events
        elif event_rate < 0.5:
            print("\n⚠️  TOO FEW COACHING EVENTS")
            print(f"  Current: {event_rate:.2f} events per video")
            print(f"  Target: 1-3 events per video")
            print("\n  Recommendations:")
            print(f"    1. Decrease MIN_PERSISTENCE_RATE from {self.current_config.MIN_PERSISTENCE_RATE} to 0.25")
            print(f"    2. Decrease MIN_DURATION_SECONDS from {self.current_config.MIN_DURATION_SECONDS} to 2.5")
            print(f"    3. Decrease MIN_CONFIDENCE from {self.current_config.MIN_CONFIDENCE} to 0.30")
            
            recommendations.append({
                'MIN_PERSISTENCE_RATE': 0.25,
                'MIN_DURATION_SECONDS': 2.5,
                'MIN_CONFIDENCE': 0.30
            })
        
        else:
            print("\n✅ Event rate looks good!")
            print(f"  {event_rate:.2f} events per video is within target range (1-3)")
        
        # Check cache hit rate
        if cache_hit_rate < 50:
            print("\n⚠️  LOW CACHE HIT RATE")
            print(f"  Current: {cache_hit_rate:.1f}%")
            print(f"  Target: 60-80%")
            print("\n  Recommendations:")
            print("    1. Add more patterns to Tier 1 cache")
            print("    2. Review top mistake types and add to cache:")
            
            # Show top uncached mistakes
            sorted_mistakes = sorted(self.results['mistake_types'].items(), 
                                   key=lambda x: x[1], reverse=True)[:10]
            for mistake, count in sorted_mistakes:
                print(f"       - {mistake}: {count} occurrences")
        
        elif cache_hit_rate > 85:
            print("\n✅ Excellent cache hit rate!")
            print(f"  {cache_hit_rate:.1f}% - Most common mistakes are cached")
        
        else:
            print("\n✅ Cache hit rate is good!")
            print(f"  {cache_hit_rate:.1f}% is within target range (60-80%)")
        
        # Check video coverage
        if video_coverage < 30:
            print("\n⚠️  LOW VIDEO COVERAGE")
            print(f"  Only {video_coverage:.1f}% of videos generated coaching events")
            print("\n  This might indicate:")
            print("    1. Thresholds are too strict")
            print("    2. Many videos have good form (expected)")
            print("    3. CV model not detecting mistakes")
            print("\n  Consider:")
            print("    - Review videos with no events manually")
            print("    - Lower thresholds if mistakes are being missed")
        
        # Generate tuned config file
        if recommendations:
            self.generate_tuned_config(recommendations[0])
    
    def generate_tuned_config(self, new_params: dict):
        """
        Generate a tuned config file
        """
        
        tuned_config = {
            'MIN_PERSISTENCE_RATE': new_params.get('MIN_PERSISTENCE_RATE', self.current_config.MIN_PERSISTENCE_RATE),
            'MIN_CONFIDENCE': new_params.get('MIN_CONFIDENCE', self.current_config.MIN_CONFIDENCE),
            'MIN_DURATION_SECONDS': new_params.get('MIN_DURATION_SECONDS', self.current_config.MIN_DURATION_SECONDS),
            'MIN_COACHING_INTERVAL': self.current_config.MIN_COACHING_INTERVAL,
            'RE_COACHING_THRESHOLD': self.current_config.RE_COACHING_THRESHOLD
        }
        
        output_file = Path("tuned_config.json")
        with open(output_file, 'w') as f:
            json.dump(tuned_config, f, indent=2)
        
        print(f"\n💾 Tuned config saved to: {output_file}")
        print("\nTest with tuned config:")
        print(f"  python test_integration.py --dataset test --config {output_file}")
    
    def compare_videos(self):
        """
        Compare videos with many events vs no events
        """
        
        print("\n" + "="*60)
        print("VIDEO ANALYSIS")
        print("="*60)
        
        videos = self.results['video_details']
        
        # Videos with most events
        videos_with_events = [v for v in videos if v['num_events'] > 0]
        videos_no_events = [v for v in videos if v['num_events'] == 0]
        
        print(f"\n📹 Videos with Events: {len(videos_with_events)}")
        print(f"⚪ Videos without Events: {len(videos_no_events)}")
        
        if videos_with_events:
            print("\nTop 10 Videos with Most Events:")
            sorted_videos = sorted(videos_with_events, key=lambda x: x['num_events'], reverse=True)[:10]
            for video in sorted_videos:
                print(f"  {video['filename']}: {video['num_events']} events")
                # Show event details
                for event in video['events'][:3]:  # Show first 3 events
                    print(f"    - {event['mistake']} (severity: {event['severity']}, tier: {event['tier']})")
        
        if videos_no_events:
            print(f"\nSample Videos with No Events (first 10):")
            for video in videos_no_events[:10]:
                print(f"  {video['filename']}")
            
            print("\n  💡 Tip: Review these videos manually to determine if:")
            print("     - They have good form (expected)")
            print("     - Thresholds are too strict (need tuning)")
            print("     - CV model isn't detecting mistakes")


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Tune Integration Layer Thresholds')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to test results JSON file')
    
    args = parser.parse_args()
    
    results_file = Path(args.results)
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        print("\nAvailable results files:")
        results_dir = Path("./test_results")
        if results_dir.exists():
            for f in sorted(results_dir.glob("*.json")):
                print(f"  {f}")
        return
    
    tuner = ThresholdTuner(results_file)
    tuner.analyze()
    tuner.compare_videos()


if __name__ == "__main__":
    main()


# ==========================================
# USAGE EXAMPLES
# ==========================================

"""
USAGE:

# 1. Analyze test results and get recommendations
python test_tune_thresholds.py --results test_results/test_results_test_20240305_143022.json

# 2. This will:
#    - Analyze current performance
#    - Identify issues (too many/few events, low cache hit rate)
#    - Generate recommendations
#    - Create tuned_config.json with suggested parameters

# 3. Test with tuned config:
python test_integration.py --dataset test --config tuned_config.json
"""
