"""
Generate visualization video for rally.mp4 analysis.
"""
from pathlib import Path
from examples.comprehensive_video_test import test_video_analysis, visualize_analysis

# Analyze video
video_path = Path('data/rally.mp4')
print("Analyzing video...")
results = test_video_analysis(video_path)

# Generate visualization
output_path = Path('data/rally_annotated.mp4')
print(f"\n{'='*80}")
print("GENERATING ANNOTATED VIDEO")
print(f"{'='*80}\n")
print(f"Output: {output_path}")
print("This will process all frames and create an annotated video...")
print("Please wait...\n")

visualize_analysis(video_path, results, output_path, max_frames=324)

print(f"\n{'='*80}")
print("✅ VISUALIZATION COMPLETE!")
print(f"{'='*80}")
print(f"\n📹 Annotated video saved to: {output_path}")
print(f"   Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
print(f"\nYou can play it with: open {output_path}")
