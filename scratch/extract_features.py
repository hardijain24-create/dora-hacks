import os, json
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

FRAME_STRIDE = 2
SEQ_LEN = 40

POSE_MODEL_PATH = 'C:/Users/Admin/Downloads/silent-interpreter/isl_tfjs_export (1)/pose_landmarker_lite.task'
HAND_MODEL_PATH = 'C:/Users/Admin/Downloads/silent-interpreter/isl_tfjs_export (1)/hand_landmarker.task'

# Download task files if they don't exist
if not os.path.exists(POSE_MODEL_PATH):
    import urllib.request
    urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task', POSE_MODEL_PATH)
if not os.path.exists(HAND_MODEL_PATH):
    import urllib.request
    urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task', HAND_MODEL_PATH)


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
_worker_pose = mp_vision.PoseLandmarker.create_from_options(pose_options)
_worker_hand = mp_vision.HandLandmarker.create_from_options(hand_options)

def _landmarks_to_array(landmark_list, n_points, n_dims):
    if not landmark_list:
        return np.zeros(n_points * n_dims)
    vals = []
    for lm in landmark_list:
        if n_dims == 4:
            vals.extend([lm.x, lm.y, lm.z, getattr(lm, 'visibility', 0.0)])
        else:
            vals.extend([lm.x, lm.y, lm.z])
    return np.array(vals)

def resample_sequence(seq, target_len=SEQ_LEN):
    n = len(seq)
    if n == target_len:
        return seq
    idx = np.linspace(0, n - 1, target_len)
    return seq[np.round(idx).astype(int)]

def normalize_sequence(seq):
    seq = seq.copy()
    pose = seq[:, :132].reshape(-1, 33, 4)
    lh = seq[:, 132:195].reshape(-1, 21, 3)
    rh = seq[:, 195:258].reshape(-1, 21, 3)

    left_sh, right_sh = pose[:, 11, :2], pose[:, 12, :2]
    center = (left_sh + right_sh) / 2.0
    scale = np.linalg.norm(left_sh - right_sh, axis=1, keepdims=True)
    scale = np.where(scale < 1e-4, 1.0, scale)

    pose[:, :, :2] -= center[:, None, :]
    pose[:, :, :2] /= scale[:, None, :]
    lh[:, :, :2] -= center[:, None, :]
    lh[:, :, :2] /= scale[:, None, :]
    rh[:, :, :2] -= center[:, None, :]
    rh[:, :, :2] /= scale[:, None, :]

    return np.concatenate([
        pose.reshape(-1, 132), lh.reshape(-1, 63), rh.reshape(-1, 63)
    ], axis=1)

def extract_one(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if i % FRAME_STRIDE == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            pose_result = _worker_pose.detect(mp_image)
            hand_result = _worker_hand.detect(mp_image)

            pose = np.zeros(33 * 4)
            if pose_result.pose_landmarks:
                pose = _landmarks_to_array(pose_result.pose_landmarks[0], 33, 4)

            lh = np.zeros(21 * 3)
            rh = np.zeros(21 * 3)
            if hand_result.hand_landmarks:
                for hand_lms, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                    label = handedness[0].category_name
                    arr = _landmarks_to_array(hand_lms, 21, 3)
                    if label == 'Left':
                        lh = arr
                    else:
                        rh = arr

            frames.append(np.concatenate([pose, lh, rh]))
        i += 1
    cap.release()
    
    if len(frames) > 0:
        seq = np.array(frames, dtype=np.float32)
        seq = normalize_sequence(seq)
        seq = resample_sequence(seq)
        return seq
    return None

import sys
import glob

if __name__ == '__main__':
    video_path = sys.argv[1]
    
    # Process multiple videos if directory
    if os.path.isdir(video_path):
        results = {}
        # Search all subdirectories for .mp4
        for root, _, files in os.walk(video_path):
            for file in files:
                if file.lower().endswith(('.mp4', '.mov', '.avi')):
                    path = os.path.join(root, file)
                    print(f"Extracting {path}...")
                    seq = extract_one(path)
                    if seq is not None:
                        # Find the parent folder name which is the label
                        label = os.path.basename(os.path.dirname(path))
                        # Use filename as key
                        key = f"{label}/{file}"
                        results[key] = seq.tolist()
        
        with open('scratch/test_sequences.json', 'w') as f:
            json.dump(results, f)
        print("Done! Wrote to scratch/test_sequences.json")
    else:
        seq = extract_one(video_path)
        if seq is not None:
            with open('scratch/test_sequence.json', 'w') as f:
                json.dump(seq.tolist(), f)
            print("Done! Wrote to scratch/test_sequence.json")
        else:
            print("Failed to extract features.")
