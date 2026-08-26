import os
import json
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import time

# --- CONFIGURATION ---
VIDEO_DIR = 'isl_reference_videos'
POSE_MODEL_PATH = 'isl_tfjs_export (1)/pose_landmarker_lite.task'
HAND_MODEL_PATH = 'isl_tfjs_export (1)/hand_landmarker.task'
OUT_DIR = 'public/models/isl-mvp'
FEATURE_OUT_DIR = 'data/mvp_features'

SEQ_LEN = 40
FEATURE_DIM = 258

MOTION_THRESHOLD = 0.005 # Min movement of keypoints to count as "moving"
MIN_GESTURE_FRAMES = 15
MAX_GESTURE_FRAMES = 150
REST_PATIENCE = 10 # Frames of rest before gesture is considered ended

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FEATURE_OUT_DIR, exist_ok=True)

# --- MEDIAPIPE INITIALIZATION ---
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
worker_pose = mp_vision.PoseLandmarker.create_from_options(pose_options)
worker_hand = mp_vision.HandLandmarker.create_from_options(hand_options)

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
        return np.zeros((target_len, FEATURE_DIM))
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

def extract_frame_features(frame_rgb):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    pose_result = worker_pose.detect(mp_image)
    hand_result = worker_hand.detect(mp_image)

    pose = np.zeros(33 * 4)
    has_person = False
    if pose_result.pose_landmarks:
        pose = _landmarks_to_array(pose_result.pose_landmarks[0], 33, 4)
        has_person = True

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

    features = np.concatenate([pose, lh, rh])
    return features, has_person

def compute_motion(prev_feat, curr_feat):
    if prev_feat is None or curr_feat is None:
        return 0.0
    
    # Check specifically hands to see if there is movement
    prev_hands = prev_feat[132:258]
    curr_hands = curr_feat[132:258]
    
    if np.sum(prev_hands) == 0 or np.sum(curr_hands) == 0:
        return 0.0
        
    diff = curr_hands - prev_hands
    motion = np.linalg.norm(diff)
    return motion

def main():
    print("Starting Phase 1-8 Pipeline...")
    start_time = time.time()
    
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4') and 'test' not in f.lower()]
    
    report = {
        "total_videos": len(video_files),
        "usable_videos": 0,
        "unusable_videos": 0,
        "total_candidate_segments": 0,
        "verified_segments": 0,
        "pending_segments": 0,
        "rejected_segments": 0,
        "total_final_samples": 0,
        "number_of_gesture_classes": 0,
        "samples_per_class": {},
        "minimum_samples_per_class": 0,
        "maximum_samples_per_class": 0,
        "average_samples_per_class": 0,
        "landmark_detection_success_rate": 0.0,
        "average_gesture_duration": 0.0,
        "minimum_original_frame_count": 9999,
        "maximum_original_frame_count": 0,
        "final_tensor_shape": None,
        "label_distribution": {},
        "video_stats": []
    }
    
    all_segments = []
    
    # Sort files to be deterministic
    video_files.sort()
    
    for v_idx, v_file in enumerate(video_files):
        v_path = os.path.join(VIDEO_DIR, v_file)
        cap = cv2.VideoCapture(v_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\\nProcessing [{v_idx+1}/{len(video_files)}] {v_file} ({frame_count} frames, {duration:.1f}s)")
        
        first_meaningful_frame = -1
        last_meaningful_frame = -1
        
        features_list = []
        is_moving = False
        current_segment_start = -1
        rest_counter = 0
        prev_feat = None
        
        i = 0
        # For a full scale run, this can take a long time, but we will process all frames.
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: break
            
            # Since MediaPipe takes time, logging progress
            if i % 100 == 0:
                print(f"  ...processed {i}/{frame_count} frames")
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feat, has_person = extract_frame_features(frame_rgb)
            
            features_list.append(feat)
            
            if has_person:
                if first_meaningful_frame == -1:
                    first_meaningful_frame = i
                last_meaningful_frame = i
            
            i += 1
                
        cap.release()
        
        if first_meaningful_frame == -1:
            print(f"  -> No person detected in {v_file}. Skipping.")
            report["unusable_videos"] += 1
            report["video_stats"].append({"video": v_file, "status": "unusable", "reason": "No person detected"})
            continue
            
        report["usable_videos"] += 1
        usable_frames = last_meaningful_frame - first_meaningful_frame + 1
        print(f"  -> Usable frames: {first_meaningful_frame} to {last_meaningful_frame} ({usable_frames} frames)")
        
        candidate_segments = []
        
        for idx in range(first_meaningful_frame, last_meaningful_frame + 1):
            curr_feat = features_list[idx]
            motion = compute_motion(prev_feat, curr_feat)
            
            if motion > MOTION_THRESHOLD:
                if not is_moving:
                    is_moving = True
                    current_segment_start = idx
                rest_counter = 0
            else:
                if is_moving:
                    rest_counter += 1
                    if rest_counter >= REST_PATIENCE:
                        is_moving = False
                        end_idx = idx - REST_PATIENCE
                        seg_len = end_idx - current_segment_start
                        if MIN_GESTURE_FRAMES <= seg_len <= MAX_GESTURE_FRAMES:
                            candidate_segments.append({
                                "video": v_file,
                                "startFrame": current_segment_start,
                                "endFrame": end_idx,
                                "startTime": current_segment_start / fps,
                                "endTime": end_idx / fps,
                                "frameCount": seg_len,
                                "label": "PENDING_REVIEW"
                            })
                        current_segment_start = -1
                        
            prev_feat = curr_feat
            
        print(f"  -> Found {len(candidate_segments)} candidate segments.")
        report["total_candidate_segments"] += len(candidate_segments)
        
        for seg in candidate_segments:
            seg_feat = np.array(features_list[seg["startFrame"]:seg["endFrame"]])
            
            total_frames = len(seg_feat)
            valid_frames = sum(1 for f in seg_feat if np.sum(f) > 0)
            valid_pct = valid_frames / total_frames if total_frames > 0 else 0
            
            if valid_pct < 0.5:
                seg["status"] = "REJECTED"
                seg["reject_reason"] = f"Low valid landmark coverage ({valid_pct:.1%})"
                report["rejected_segments"] += 1
            else:
                seg["status"] = "PENDING"
                report["pending_segments"] += 1
                
                norm_feat = normalize_sequence(seg_feat)
                resampled = resample_sequence(norm_feat, SEQ_LEN)
                seg["sequence"] = resampled.tolist()
                
                report["average_gesture_duration"] += total_frames
                report["minimum_original_frame_count"] = min(report["minimum_original_frame_count"], total_frames)
                report["maximum_original_frame_count"] = max(report["maximum_original_frame_count"], total_frames)
            
            all_segments.append(seg)
            
        report["video_stats"].append({
            "video": v_file,
            "status": "usable",
            "first_frame": first_meaningful_frame,
            "last_frame": last_meaningful_frame,
            "candidates": len(candidate_segments)
        })

    valid_samples = [s for s in all_segments if s.get("status") == "PENDING"]
    report["total_final_samples"] = len(valid_samples)
    
    if report["total_final_samples"] > 0:
        report["average_gesture_duration"] /= report["total_final_samples"]
        report["final_tensor_shape"] = [report["total_final_samples"], SEQ_LEN, FEATURE_DIM]
    else:
        report["minimum_original_frame_count"] = 0
        
    print("\\nSaving outputs...")
    
    review_segments = []
    for s in all_segments:
        s_copy = s.copy()
        if "sequence" in s_copy:
            del s_copy["sequence"]
        review_segments.append(s_copy)
        
    with open(os.path.join(OUT_DIR, 'mvp_segments.json'), 'w') as f:
        json.dump(review_segments, f, indent=2)
        
    with open(os.path.join(OUT_DIR, 'dataset_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
        
    if valid_samples:
        X = np.array([s["sequence"] for s in valid_samples])
        y = np.array([s["label"] for s in valid_samples])
        
        np.save(os.path.join(FEATURE_OUT_DIR, 'X.npy'), X)
        np.save(os.path.join(FEATURE_OUT_DIR, 'y.npy'), y)
        
        dataset_json = []
        for s in valid_samples:
            dataset_json.append({
                "label": s["label"],
                "sequence": s["sequence"],
                "source": f"{s['video']}@{s['startTime']:.1f}-{s['endTime']:.1f}"
            })
            
        with open(os.path.join(OUT_DIR, 'dataset.json'), 'w') as f:
            json.dump(dataset_json, f)
            
    print(f"Pipeline completed in {time.time() - start_time:.1f} seconds.")
    print("Report generated at", os.path.join(OUT_DIR, 'dataset_report.json'))

if __name__ == '__main__':
    main()
