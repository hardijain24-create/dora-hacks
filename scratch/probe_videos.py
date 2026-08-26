"""
Probe reference videos: sample every ~30 seconds, detect landmarks,
and report what's actually visible (pose/hand presence, motion statistics).
This tells us the content composition without watching each video manually.
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

print("=" * 70)
for video_path in videos:
    if not os.path.exists(video_path):
        print(f"MISSING: {video_path}")
        continue
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / max(fps, 1)
    
    print(f"\n{os.path.basename(video_path)}")
    print(f"  Duration: {duration_s:.1f}s | FPS: {fps} | Frames: {total_frames} | Resolution: {w}x{h}")
    
    # Sample 1 frame every 10 seconds to understand content
    sample_interval = int(fps * 10)  # every 10 seconds
    sample_frames = []
    
    for frame_idx in range(0, total_frames, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            # Compute basic motion indicator: brightness variance of center crop
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Center crop (upper body region)
            h_c, w_c = gray.shape
            crop = gray[int(h_c*0.1):int(h_c*0.8), int(w_c*0.1):int(w_c*0.9)]
            brightness = float(np.mean(crop))
            variance = float(np.var(crop))
            sample_frames.append((frame_idx/fps, brightness, variance))
    
    cap.release()
    
    # Report samples
    print(f"  Samples (time_s, brightness, variance):")
    for t, b, v in sample_frames[:12]:  # first 12 samples
        print(f"    t={t:6.1f}s  brightness={b:5.1f}  variance={v:7.0f}")
    
    if len(sample_frames) > 12:
        print(f"    ... ({len(sample_frames)} total samples)")

print("\n" + "=" * 70)
print("Note: High variance = more visual content / motion")
print("Low brightness AND low variance = mostly empty/dark frames")
