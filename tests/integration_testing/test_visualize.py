"""
Visualization Script for Integration Layer Test Results
Generates charts and graphs to analyze test performance

Usage:
    python test_visualize.py --results test_results_test_20240305_143022.json
"""

import json
import argparse
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


# ==========================================
# VISUALIZATION GENERATOR
# ==========================================

class TestVisualizer:
    """
    Generate visualizations from test results
    """
    
    def __init__(self, results_file: Path):
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        self.output_dir = Path("./test_results/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get dataset name from filename
        self.dataset_name = results_file.stem.replace('test_results_', '')
    
    def generate_all(self):
        """
        Generate all visualizations
        """
        
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        try:
            self.plot_tier_distribution()
            self.plot_severity_distribution()
            self.plot_top_mistakes()
            self.plot_top_exercises()
            self.plot_events_per_video_distribution()
            self.plot_mistake_severity_heatmap()
            
            print(f"\n✅ All visualizations saved to: {self.output_dir}")
            
        except Exception as e:
            print(f"\n❌ Error generating visualizations: {e}")
            print("Make sure matplotlib is installed: pip install matplotlib")
    
    def plot_tier_distribution(self):
        """
        Plot tier distribution (Tier 1/2/3)
        """
        
        tiers = ['Tier 1\n(Cache)', 'Tier 2\n(RAG)', 'Tier 3\n(Reasoning)']
        counts = [
            self.results['tier_breakdown']['tier_1'],
            self.results['tier_breakdown']['tier_2'],
            self.results['tier_breakdown']['tier_3']
        ]
        
        colors = ['#43e97b', '#4facfe', '#f5576c']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(tiers, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Number of Events', fontsize=12, fontweight='bold')
        ax.set_title('Tier Distribution: Routing Decisions', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add cache hit rate annotation
        total = sum(counts)
        if total > 0:
            cache_hit_rate = counts[0] / total * 100
            ax.text(0.98, 0.98, f'Cache Hit Rate: {cache_hit_rate:.1f}%',
                   transform=ax.transAxes,
                   ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        output_file = self.output_dir / f"{self.dataset_name}_tier_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Tier distribution: {output_file}")
    
    def plot_severity_distribution(self):
        """
        Plot severity distribution (High/Medium/Low)
        """
        
        severities = ['High', 'Medium', 'Low']
        counts = [
            self.results['severity_breakdown']['high'],
            self.results['severity_breakdown']['medium'],
            self.results['severity_breakdown']['low']
        ]
        
        colors = ['#f5576c', '#feca57', '#43e97b']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(severities, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Number of Events', fontsize=12, fontweight='bold')
        ax.set_title('Severity Distribution: Mistake Classification', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / f"{self.dataset_name}_severity_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Severity distribution: {output_file}")
    
    def plot_top_mistakes(self, top_n=15):
        """
        Plot top N most common mistakes
        """
        
        mistakes = self.results['mistake_types']
        sorted_mistakes = sorted(mistakes.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        names = [m[0] for m in sorted_mistakes]
        counts = [m[1] for m in sorted_mistakes]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(names)), counts, color='#667eea', alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel('Number of Events', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Most Common Mistakes', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count, i, f' {count}', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        output_file = self.output_dir / f"{self.dataset_name}_top_mistakes.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Top mistakes: {output_file}")
    
    def plot_top_exercises(self, top_n=15):
        """
        Plot top N exercises with most coaching events
        """
        
        exercises = self.results['exercise_types']
        sorted_exercises = sorted(exercises.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        names = [e[0] for e in sorted_exercises]
        counts = [e[1] for e in sorted_exercises]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(names)), counts, color='#43e97b', alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel('Number of Events', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Exercises with Most Coaching Events', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count, i, f' {count}', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        output_file = self.output_dir / f"{self.dataset_name}_top_exercises.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Top exercises: {output_file}")
    
    def plot_events_per_video_distribution(self):
        """
        Plot distribution of events per video
        """
        
        events_per_video = self.results['events_per_video']
        
        # Count frequency
        counter = Counter(events_per_video)
        sorted_items = sorted(counter.items())
        
        event_counts = [item[0] for item in sorted_items]
        frequencies = [item[1] for item in sorted_items]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(event_counts, frequencies, color='#4facfe', alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Number of Events per Video', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Videos', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Events per Video', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean line
        mean_events = sum(events_per_video) / len(events_per_video)
        ax.axvline(mean_events, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_events:.2f}')
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        output_file = self.output_dir / f"{self.dataset_name}_events_per_video.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Events per video: {output_file}")
    
    def plot_mistake_severity_heatmap(self):
        """
        Plot heatmap of mistake types vs severity
        """
        
        # Build matrix of mistake × severity
        mistake_severity = {}
        
        for video in self.results['video_details']:
            for event in video['events']:
                mistake = event['mistake']
                severity = event['severity']
                
                if mistake not in mistake_severity:
                    mistake_severity[mistake] = {'high': 0, 'medium': 0, 'low': 0}
                
                mistake_severity[mistake][severity] += 1
        
        # Get top mistakes
        top_mistakes = sorted(mistake_severity.items(), 
                            key=lambda x: sum(x[1].values()), 
                            reverse=True)[:20]
        
        mistakes = [m[0] for m in top_mistakes]
        data = [[m[1]['high'], m[1]['medium'], m[1]['low']] for m in top_mistakes]
        
        fig, ax = plt.subplots(figsize=(10, 12))
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
        
        # Set ticks
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['High', 'Medium', 'Low'], fontsize=11)
        ax.set_yticks(range(len(mistakes)))
        ax.set_yticklabels(mistakes, fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Events', fontsize=11, fontweight='bold')
        
        # Add text annotations
        for i in range(len(mistakes)):
            for j in range(3):
                text = ax.text(j, i, int(data[i][j]),
                             ha="center", va="center", color="black", fontsize=9, fontweight='bold')
        
        ax.set_title('Mistake Type vs Severity Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = self.output_dir / f"{self.dataset_name}_mistake_severity_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Mistake-severity heatmap: {output_file}")


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Visualize Integration Layer Test Results')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to test results JSON file')
    
    args = parser.parse_args()
    
    results_file = Path(args.results)
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    visualizer = TestVisualizer(results_file)
    visualizer.generate_all()


if __name__ == "__main__":
    main()


# ==========================================
# USAGE
# ==========================================

"""
USAGE:

# Generate all visualizations
python test_visualize.py --results test_results/test_results_test_20240305_143022.json

# This will create:
#   - Tier distribution bar chart
#   - Severity distribution bar chart
#   - Top 15 mistakes bar chart
#   - Top 15 exercises bar chart
#   - Events per video histogram
#   - Mistake vs severity heatmap

# All saved to: test_results/figures/
"""
