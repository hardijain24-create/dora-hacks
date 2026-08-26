import os
import sys
import json
import csv
import time
import argparse
import subprocess
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Configuration ─────────────────────────────────────────────────────────────
VIDEO_DIR = 'isl_reference_videos'
REVIEW_DIR = 'scratch/segment_review'
PROPOSALS_OUT = 'data/segment_proposals.json'
ANNOTATIONS_OUT = 'data/mvp_annotations.json'
CSV_OUT = 'data/segment_review.csv'
DIAG_CSV_OUT = 'data/velocity_06.csv'
BOUNDARY_CSV_OUT = 'data/boundary_diagnostics.csv'

POSE_MODEL_PATH = 'isl_tfjs_export (1)/pose_landmarker_lite.task'
HAND_MODEL_PATH = 'isl_tfjs_export (1)/hand_landmarker.task'

FRAME_STRIDE = 3
BRIGHT_THRESH = 15.0

# ── Segmentation Parameters ───────────────────────────────────────────────────
MOTION_START_THRESH = 0.05
MOTION_STOP_THRESH = 0.03
DROP_RATIO_THRESH = 0.60
VALLEY_PATIENCE_FRAMES = 3
MIN_GESTURE_SECS = 0.5
MAX_GESTURE_SECS = 5.0
PAD_SECS = 0.2
SMOOTHING_WINDOW = 5

os.makedirs('data', exist_ok=True)
os.makedirs(REVIEW_DIR, exist_ok=True)

def extract_motion_points(frame_rgb, pose_det, hand_det):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    pose_r = pose_det.detect(mp_image)
    hand_r = hand_det.detect(mp_image)
    
    pts = {}
    has_pose = False
    has_hand = False
    shoulder_dist = None
    
    if pose_r.pose_landmarks:
        has_pose = True
        lms = pose_r.pose_landmarks[0]
        if len(lms) > 16:
            pts['wrist_l'] = (lms[15].x, lms[15].y)
            pts['wrist_r'] = (lms[16].x, lms[16].y)
            
            sh_l = np.array([lms[11].x, lms[11].y])
            sh_r = np.array([lms[12].x, lms[12].y])
            dist = np.linalg.norm(sh_l - sh_r)
            if dist > 1e-4:
                shoulder_dist = float(dist)
                
    if hand_r.hand_landmarks and hand_r.handedness:
        has_hand = True
        for hand_lms, handedness in zip(hand_r.hand_landmarks, hand_r.handedness):
            label = handedness[0].category_name
            key = 'index_l' if label == 'Left' else 'index_r'
            if len(hand_lms) > 8:
                pts[key] = (hand_lms[8].x, hand_lms[8].y)
                
    return pts, has_pose, has_hand, shoulder_dist

def point_velocity(prev_pts, current_pts, shoulder_dist):
    if not prev_pts or not current_pts: return 0.0
    if not shoulder_dist: shoulder_dist = 1.0
    
    vels = []
    for k in ['wrist_l', 'wrist_r', 'index_l', 'index_r']:
        if k in prev_pts and k in current_pts:
            dx = current_pts[k][0] - prev_pts[k][0]
            dy = current_pts[k][1] - prev_pts[k][1]
            dist = ((dx*dx + dy*dy) ** 0.5) / shoulder_dist
            vels.append(dist)
            
    if not vels: return 0.0
    return sum(vels) / len(vels)

def extract_clip(video_in, start_sec, end_sec, clip_out):
    cmd = [
        'ffmpeg', '-y', '-ss', str(max(0, start_sec)), '-i', video_in,
        '-t', str(end_sec - start_sec),
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
        '-an', clip_out
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

class Segmenter:
    def __init__(self, fps):
        self.fps = fps
        self.in_gesture = False
        self.gest_start_t = 0
        self.peak_vel = 0.0
        self.peak_t = 0.0
        self.valley_count = 0
        self.current_frames = []
        
        self.raw_candidates = []
        self.boundary_diagnostics = []
        
    def emit_candidate(self, end_t, reason, valley_vel, drop_ratio):
        dur = end_t - self.gest_start_t
        
        # Enforce mathematical invariant
        if dur > MAX_GESTURE_SECS:
            end_t = self.gest_start_t + MAX_GESTURE_SECS
            dur = MAX_GESTURE_SECS
            reason = "FORCED_MAX_DUR"
            
        frames = [f for f in self.current_frames if f['t'] <= end_t]
        
        diag = {
            "peak_t": self.peak_t,
            "peak_vel": self.peak_vel,
            "valley_t": end_t,
            "valley_vel": valley_vel,
            "drop_ratio": drop_ratio,
            "valley_dur": self.valley_count,
            "cand_dur_before": dur,
            "cand_dur_after": dur,
            "reason": reason
        }
        
        if dur < MIN_GESTURE_SECS:
            diag["reason"] += " (REJECTED_MIN_DUR)"
            self.boundary_diagnostics.append(diag)
            return end_t
            
        diag["reason"] += " (ACCEPTED)"
        self.boundary_diagnostics.append(diag)
            
        self.raw_candidates.append({
            "start": self.gest_start_t,
            "end": end_t,
            "frames": frames
        })
        return end_t
        
    def process_frame(self, t, smoothed_vel, has_hand, has_pose):
        if not self.in_gesture:
            if smoothed_vel > MOTION_START_THRESH and has_hand:
                self.in_gesture = True
                self.gest_start_t = t
                self.peak_vel = smoothed_vel
                self.peak_t = t
                self.valley_count = 0
                self.current_frames = [{"t": t, "has_pose": has_pose, "has_hand": has_hand, "vel": smoothed_vel}]
        else:
            self.current_frames.append({"t": t, "has_pose": has_pose, "has_hand": has_hand, "vel": smoothed_vel})
            
            if smoothed_vel > self.peak_vel:
                self.peak_vel = smoothed_vel
                self.peak_t = t
                
            drop_ratio = (self.peak_vel - smoothed_vel) / max(self.peak_vel, 0.001)
            
            is_valley = smoothed_vel <= MOTION_STOP_THRESH or drop_ratio >= DROP_RATIO_THRESH
            
            if is_valley:
                self.valley_count += 1
            else:
                self.valley_count = 0
                
            dur = t - self.gest_start_t
            
            if self.valley_count >= VALLEY_PATIENCE_FRAMES:
                boundary_t = t - (self.valley_count * (FRAME_STRIDE / self.fps))
                if boundary_t - self.gest_start_t >= MIN_GESTURE_SECS:
                    new_start = self.emit_candidate(boundary_t, "VALLEY_DETECTED", smoothed_vel, drop_ratio)
                    
                    if smoothed_vel <= MOTION_STOP_THRESH:
                        self.in_gesture = False
                    else:
                        self.gest_start_t = new_start
                        self.peak_vel = smoothed_vel
                        self.peak_t = t
                        self.valley_count = 0
                        self.current_frames = [f for f in self.current_frames if f['t'] >= new_start]
            
            elif dur >= MAX_GESTURE_SECS:
                # Force an intelligent split on the lowest local minimum in the last N frames
                sub = self.current_frames[-int(2.0*self.fps/FRAME_STRIDE):] # last 2 seconds
                if not sub: sub = self.current_frames
                min_f = min(sub, key=lambda f: f['vel'])
                boundary_t = min_f['t']
                
                new_start = self.emit_candidate(boundary_t, "FORCED_SPLIT", min_f['vel'], drop_ratio)
                
                self.gest_start_t = new_start
                self.peak_vel = smoothed_vel
                self.peak_t = t
                self.valley_count = 0
                self.current_frames = [f for f in self.current_frames if f['t'] >= new_start]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()

    print("PHASE 1 - GENERATE SEGMENT PROPOSALS")
    
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4') and 'test' not in f.lower()]
    video_files.sort()
    
    if args.test_only:
        video_files = [f for f in video_files if '06_gestures_short' in f]
        print("TEST RUN MODE: Only processing 06_gestures_short.mp4\n")

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

    report_summary = []
    total_candidates = 0
    videos_processed = 0
    
    proposals = []
    annotations = []
    csv_rows = []
    
    for v_idx, v_file in enumerate(video_files):
        v_path = os.path.join(VIDEO_DIR, v_file)
        v_name = os.path.splitext(v_file)[0]
        
        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened(): continue
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n[{v_idx+1}/{len(video_files)}] {v_file}")
        
        intro_ended = False
        prev_pts = None
        
        raw_vel_history = []
        shoulder_history = []
        velocity_log = []
        
        segmenter = Segmenter(fps)
        
        frame_idx = 0
        start_time = time.time()
        
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: break
            frame_idx += 1
            t = frame_idx / fps
            
            if frame_idx % 100 == 0 or frame_idx == total_frames:
                pct = (frame_idx / total_frames) * 100
                elapsed = time.time() - start_time
                fps_proc = frame_idx / elapsed if elapsed > 0 else 0
                eta = (total_frames - frame_idx) / fps_proc if fps_proc > 0 else 0
                print(f"\rProcessing: {frame_idx}/{total_frames} ({pct:.1f}%) | ETA: {eta:.0f}s", end='', flush=True)

            if not intro_ended:
                if frame.mean() > BRIGHT_THRESH: intro_ended = True
                else: continue

            if frame_idx % FRAME_STRIDE != 0: continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pts, has_pose, has_hand, shoulder_dist = extract_motion_points(frame_rgb, worker_pose, worker_hand)
            
            if shoulder_dist:
                shoulder_history.append(shoulder_dist)
                if len(shoulder_history) > SMOOTHING_WINDOW: shoulder_history.pop(0)
            avg_shoulder = np.mean(shoulder_history) if shoulder_history else 1.0
            
            vel = point_velocity(prev_pts, pts, avg_shoulder)
            
            raw_vel_history.append(vel)
            if len(raw_vel_history) > SMOOTHING_WINDOW: raw_vel_history.pop(0)
            smoothed_vel = np.mean(raw_vel_history) if raw_vel_history else 0.0

            velocity_log.append({"frame_idx": frame_idx, "t": t, "vel": smoothed_vel, "hand_vis": has_hand, "state": "CAPTURING" if segmenter.in_gesture else "IDLE"})
            
            segmenter.process_frame(t, smoothed_vel, has_hand, has_pose)
            prev_pts = pts
            
        cap.release()
        videos_processed += 1
        
        vid_out_dir = os.path.join(REVIEW_DIR, v_name)
        os.makedirs(vid_out_dir, exist_ok=True)
        
        valid_candidates = 0
        segment_id_counter = 1
        
        # Apply strict clamping to padded bounds
        clamped_candidates = []
        for i, cand in enumerate(segmenter.raw_candidates):
            c_start = cand["start"] - PAD_SECS
            c_end = cand["end"] + PAD_SECS
            
            if i > 0:
                prev_end = clamped_candidates[-1]["end"]
                if c_start < prev_end:
                    mid = (cand["start"] + segmenter.raw_candidates[i-1]["end"]) / 2.0
                    clamped_candidates[-1]["end"] = round(mid, 2)
                    c_start = mid
            
            clamped_candidates.append({
                "start": max(0.0, round(c_start, 2)),
                "end": round(c_end, 2),
                "core_dur": cand["end"] - cand["start"],
                "frames": cand["frames"]
            })
            
        for cand in clamped_candidates:
            dur = cand["end"] - cand["start"]
            frames = cand["frames"]
            
            pose_vis = sum(1 for f in frames if f["has_pose"]) / max(1, len(frames))
            hand_vis = sum(1 for f in frames if f["has_hand"]) / max(1, len(frames))
            
            prop_id = f"{v_name}_{segment_id_counter:03d}"
            segment_id_counter += 1
            valid_candidates += 1
            
            prop = {
                "id": prop_id,
                "video": v_path,
                "start": cand["start"],
                "end": cand["end"],
                "duration": round(dur, 2),
                "label": "NEEDS_ANNOTATION",
                "hand_visibility": round(hand_vis, 2),
                "pose_visibility": round(pose_vis, 2),
                "status": "PROPOSED"
            }
            proposals.append(prop)
            annotations.append({"id": prop_id, "label": ""})
            
            clip_path = os.path.join(vid_out_dir, f"{prop_id}.mp4")
            extract_clip(v_path, prop["start"], prop["end"], clip_path)
            
            csv_rows.append([
                prop_id, v_file, prop["start"], prop["end"], prop["duration"],
                "NEEDS_ANNOTATION", clip_path
            ])
            
        if args.test_only:
            with open(DIAG_CSV_OUT, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["frame_idx", "timestamp", "smoothed_vel", "hand_vis", "state"])
                for d in velocity_log:
                    writer.writerow([d["frame_idx"], d["t"], d["vel"], d["hand_vis"], d["state"]])
            with open(BOUNDARY_CSV_OUT, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["peak_t", "peak_vel", "valley_t", "valley_vel", "drop_ratio", "valley_dur", "cand_dur_before", "cand_dur_after", "reason"])
                writer.writeheader()
                writer.writerows(segmenter.boundary_diagnostics)
                
        total_candidates += valid_candidates
        report_summary.append(f"{v_file} -> {valid_candidates} candidates")
        print(f"\nExtracted {valid_candidates} valid clips")

    with open(PROPOSALS_OUT, 'w') as f: json.dump(proposals, f, indent=2)
    with open(ANNOTATIONS_OUT, 'w') as f: json.dump(annotations, f, indent=2)
    with open(CSV_OUT, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Video", "Start", "End", "Duration", "Current Label", "Suggested Review"])
        writer.writerows(csv_rows)
        
    print("\n========================================")
    print("SEGMENTATION COMPLETE")
    for rep in report_summary:
        print(f"  {rep}")

if __name__ == '__main__':
    main()
