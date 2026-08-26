"""
Quick test: run MediaPipe landmark extraction on first 60 frames of 06_gestures_short.mp4
to confirm Python extraction pipeline works correctly.
"""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

POSE_MODEL_PATH = 'isl_tfjs_export (1)/pose_landmarker_lite.task'
HAND_MODEL_PATH = 'isl_tfjs_export (1)/hand_landmarker.task'

BaseOptions = mp_python.BaseOptions
VisionRunningMode = mp_vision.RunningMode

pose_options = mp_vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=1,
)
hand_options = mp_vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
)
pose_det = mp_vision.PoseLandmarker.create_from_options(pose_options)
hand_det = mp_vision.HandLandmarker.create_from_options(hand_options)

cap = cv2.VideoCapture('isl_reference_videos/06_gestures_short.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}, testing first 90 frames (~3s)")

frames_with_pose = 0
frames_with_hand = 0
total = 0

for i in range(90):
    ok, frame = cap.read()
    if not ok:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    pose_result = pose_det.detect(mp_image)
    hand_result = hand_det.detect(mp_image)
    
    has_pose = bool(pose_result.pose_landmarks)
    has_hand = bool(hand_result.hand_landmarks)
    total += 1
    if has_pose:
        frames_with_pose += 1
    if has_hand:
        frames_with_hand += 1
    
    if i % 15 == 0:
        print(f"  Frame {i:3d}: pose={'YES' if has_pose else ' NO'}, hands={'YES' if has_hand else ' NO'}")

cap.release()
print(f"\nSummary over {total} frames:")
print(f"  Pose detected: {frames_with_pose}/{total} ({100*frames_with_pose//total}%)")
print(f"  Hands detected: {frames_with_hand}/{total} ({100*frames_with_hand//total}%)")
print("\nPython MediaPipe extraction: WORKING" if frames_with_pose > 0 else "\nWARNING: No pose detected")
