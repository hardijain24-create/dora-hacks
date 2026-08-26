import os
import json
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

FRAME_STRIDE = 1
SEQ_LEN = 40
FEATURE_DIM = 258

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
    if n == 0:
        return np.zeros((target_len, 258))
    if n == target_len:
        return seq
    idx = np.linspace(0, n - 1, target_len)
    return seq[np.round(idx).astype(int)]

def normalize_sequence(seq):
    if len(seq) == 0:
        return seq
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

def extract_segment(video_path, start_sec, end_sec):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps) if end_sec > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    i = start_frame
    while cap.isOpened() and i <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
            
        if (i - start_frame) % FRAME_STRIDE == 0:
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

if __name__ == '__main__':
    manifest_path = 'data/isl_reference_manifest.json'
    out_dir = 'public/models/isl-mvp'
    os.makedirs(out_dir, exist_ok=True)
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    all_data = []
    
    for entry in manifest:
        video_path = entry['video']
        if not os.path.exists(video_path):
            print(f"Warning: Video not found {video_path}")
            continue
            
        print(f"Processing {video_path}...")
        for seg in entry['segments']:
            start_sec = seg.get('start', 0.0)
            end_sec = seg.get('end', 0.0)
            label = seg.get('label', 'UNKNOWN')
                
            seq = extract_segment(video_path, start_sec, end_sec)
            if seq is not None:
                # Store as list for JSON serialization
                all_data.append({
                    "label": label,
                    "sequence": seq.tolist(),
                    "source": video_path
                })
                
    if len(all_data) > 0:
        # Save combined dataset to public/models/isl-mvp/dataset.json
        out_file = os.path.join(out_dir, 'dataset.json')
        with open(out_file, 'w') as f:
            json.dump(all_data, f)
        print(f"Extraction complete! Saved {len(all_data)} sequences to {out_file}.")
    else:
        print("No valid segments found.")
