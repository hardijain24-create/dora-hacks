"""
Scan ALL reference videos for first visible (non-black) frame
and compute overall content brightness distribution.
"""
import cv2
import numpy as np
import os

videos = [
    'isl_reference_videos/01_alphabet.mp4',
    'isl_reference_videos/02_gestures.mp4',
    'isl_reference_videos/03_gestures.mp4',
    'isl_reference_videos/04_gestures.mp4',
    'isl_reference_videos/06_gestures_short.mp4',
]

BRIGHT_THRESH = 10.0  # mean pixel value above which frame is "visible"
SAMPLE_EVERY = 30     # sample every Nth frame for speed

for video_path in videos:
    if not os.path.exists(video_path):
        print(f"MISSING: {video_path}\n")
        continue
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / max(fps, 1)
    
    first_bright_frame = None
    bright_segments = []
    in_bright = False
    seg_start = None
    
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        if i % SAMPLE_EVERY == 0:
            mean_b = float(frame.mean())
            is_bright = mean_b > BRIGHT_THRESH
            
            if is_bright and first_bright_frame is None:
                first_bright_frame = (i, i/fps, mean_b)
            
            if is_bright and not in_bright:
                seg_start = i/fps
                in_bright = True
            elif not is_bright and in_bright:
                bright_segments.append((seg_start, i/fps))
                in_bright = False
        
        i += 1
    
    if in_bright:
        bright_segments.append((seg_start, duration))
    
    cap.release()
    
    name = os.path.basename(video_path)
    print(f"\n{'='*60}")
    print(f"{name}: {total} frames, {fps:.0f}fps, {duration:.0f}s total")
    
    if first_bright_frame is None:
        print("  !! ENTIRE VIDEO IS BLACK/INVISIBLE")
    else:
        fi, ft, fb = first_bright_frame
        print(f"  First visible frame: #{fi} at t={ft:.1f}s (brightness={fb:.1f})")
    
    if bright_segments:
        print(f"  Visible content segments (sampled every {SAMPLE_EVERY} frames):")
        total_visible = sum(e-s for s, e in bright_segments)
        for s, e in bright_segments[:5]:
            print(f"    {s:.1f}s – {e:.1f}s  ({e-s:.1f}s)")
        if len(bright_segments) > 5:
            print(f"    ... ({len(bright_segments)} segments total, {total_visible:.0f}s visible)")
    else:
        print("  No visible content segments found")

print("\n" + "="*60)
print("DONE")
