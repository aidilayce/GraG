#!/usr/bin/env python3
"""
Convert MP4 videos to optimized GIFs for GitHub README display.
This script creates smaller, optimized GIFs suitable for web display.
"""
import os
import sys
import imageio
from pathlib import Path

def convert_mp4_to_gif(mp4_path, gif_path, max_frames=150, fps=10, scale=0.5):
    """
    Convert MP4 to optimized GIF.
    
    Args:
        mp4_path: Path to input MP4 file
        gif_path: Path to output GIF file
        max_frames: Maximum number of frames to include (for file size)
        fps: Frames per second for output GIF
        scale: Scale factor (0.5 = half size)
    """
    print(f"Converting {mp4_path} to {gif_path}...")
    
    try:
        # Read video
        reader = imageio.get_reader(mp4_path)
        fps_original = reader.get_meta_data()['fps']
        
        # Calculate frame skip to get desired fps
        frame_skip = max(1, int(fps_original / fps))
        
        frames = []
        frame_count = 0
        
        for i, frame in enumerate(reader):
            if i % frame_skip == 0 and frame_count < max_frames:
                # Resize frame
                import numpy as np
                from PIL import Image
                img = Image.fromarray(frame)
                new_size = (int(img.width * scale), int(img.height * scale))
                img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
                frames.append(np.array(img_resized))
                frame_count += 1
        
        reader.close()
        
        # Write GIF
        imageio.mimsave(gif_path, frames, fps=fps, loop=0)
        print(f"  ✓ Created {gif_path} ({frame_count} frames, {fps} fps)")
        
        # Get file sizes
        mp4_size = os.path.getsize(mp4_path) / (1024 * 1024)  # MB
        gif_size = os.path.getsize(gif_path) / (1024 * 1024)  # MB
        print(f"  Size: {mp4_size:.1f} MB → {gif_size:.1f} MB")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    return True

def main():
    assets_dir = Path(__file__).parent.parent / "assets"
    
    videos = [
        "droid_result.mp4",
        "vipe_result.mp4",
        "handedness1.mp4",
        "handedness2.mp4",
        "handedness3.mp4",
        "handedness4.mp4",
    ]
    
    print("Converting videos to GIFs for GitHub README...")
    print("=" * 60)
    
    for video in videos:
        mp4_path = assets_dir / video
        gif_path = assets_dir / video.replace(".mp4", ".gif")
        
        if not mp4_path.exists():
            print(f"⚠ Skipping {video} (not found)")
            continue
        
        convert_mp4_to_gif(mp4_path, gif_path, max_frames=1000, fps=20, scale=0.6)
        print()
    
    print("=" * 60)
    print("Done! Update README.md to use .gif files instead of .mp4")

if __name__ == "__main__":
    main()

