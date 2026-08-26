import os
import json
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

VIDEO_PATH = 'isl_reference_videos/06_gestures_short.mp4'
POSE_MODEL_PATH = 'isl_tfjs_export (1)/pose_landmarker_lite.task'
HAND_MODEL_PATH = 'isl_tfjs_export (1)/hand_landmarker.task'

os.makedirs('data', exist_ok=True)
os.makedirs('scratch/contact_sheet', exist_ok=True)

def main():
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

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    frame_idx = 0
    data_rows = []
    
    prev_pts = {}
    
    # We will sample approximately every 0.5 seconds, but wait, the prompt says:
    # "Sample approximately every 0.5 seconds from 3.0s to the end" for the visual report
    # and "Generate a CSV and a simple diagnostic plot containing at minimum: timestamp, left wrist velocity..."
    # Actually I should just run on every stride frame (every 3rd frame) so we get the exact same signal as the original script, but parse it properly.
    
    
    # Contact sheet sampling (every 0.5 seconds = 15 frames)
    if frame_idx >= 3.0 * fps and frame_idx % int(0.5 * fps) == 0:
        cv2.imwrite(f'scratch/contact_sheet/sample_{t:.1f}s.jpg', frame)
        
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1
        t = frame_idx / fps
        
        if frame_idx % 3 != 0: continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        pose_r = worker_pose.detect(mp_image)
        hand_r = worker_hand.detect(mp_image)
        
        current_pts = {}
        has_pose = False
        has_left_hand = False
        has_right_hand = False
        
        if pose_r.pose_landmarks:
            has_pose = True
            lms = pose_r.pose_landmarks[0]
            if len(lms) > 16:
                current_pts['wrist_l'] = (lms[15].x, lms[15].y)
                current_pts['wrist_r'] = (lms[16].x, lms[16].y)
                
        if hand_r.hand_landmarks and hand_r.handedness:
            for hand_lms, handedness in zip(hand_r.hand_landmarks, hand_r.handedness):
                label = handedness[0].category_name
                # Note: MediaPipe image mirroring often flips Left/Right.
                key = 'index_l' if label == 'Left' else 'index_r'
                if label == 'Left': has_left_hand = True
                else: has_right_hand = True
                
                if len(hand_lms) > 8:
                    current_pts[key] = (hand_lms[8].x, hand_lms[8].y)
                    
        vels = {}
        for k in ['wrist_l', 'wrist_r', 'index_l', 'index_r']:
            if k in current_pts and k in prev_pts:
                dx = current_pts[k][0] - prev_pts[k][0]
                dy = current_pts[k][1] - prev_pts[k][1]
                vels[k] = (dx*dx + dy*dy) ** 0.5
            else:
                vels[k] = 0.0
                
        # To emulate the buggy script's behavior but safely:
        # Actually I just want to plot the individual velocities to see if one spikes.
        avg_vel = sum(vels.values()) / max(1, len([v for v in vels.values() if v > 0])) if any(v > 0 for v in vels.values()) else 0.0
        
        data_rows.append({
            't': t,
            'pose': has_pose,
            'hand_l': has_left_hand,
            'hand_r': has_right_hand,
            'v_wrist_l': vels.get('wrist_l', 0.0),
            'v_wrist_r': vels.get('wrist_r', 0.0),
            'v_index_l': vels.get('index_l', 0.0),
            'v_index_r': vels.get('index_r', 0.0),
            'v_avg': avg_vel
        })
        
        # Contact sheet sampling (every 0.5 seconds = 15 frames)
        if frame_idx >= 3.0 * fps and frame_idx % int(0.5 * fps) == 0:
            # draw points and save
            sheet_frame = frame.copy()
            if 'wrist_l' in current_pts:
                cv2.circle(sheet_frame, (int(current_pts['wrist_l'][0]*sheet_frame.shape[1]), int(current_pts['wrist_l'][1]*sheet_frame.shape[0])), 5, (0,255,0), -1)
            cv2.imwrite(f'scratch/contact_sheet/sample_{t:.1f}s.jpg', sheet_frame)
            
        prev_pts = current_pts

    cap.release()
    
    # Save CSV
    with open('data/velocity_06_detailed.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data_rows[0].keys())
        writer.writeheader()
        writer.writerows(data_rows)
        
    # Plot
    ts = [r['t'] for r in data_rows]
    v_wl = [r['v_wrist_l'] for r in data_rows]
    v_wr = [r['v_wrist_r'] for r in data_rows]
    v_il = [r['v_index_l'] for r in data_rows]
    v_ir = [r['v_index_r'] for r in data_rows]
    v_avg = [r['v_avg'] for r in data_rows]
    
    plt.figure(figsize=(15, 6))
    plt.plot(ts, v_wl, label='Wrist L', alpha=0.5)
    plt.plot(ts, v_wr, label='Wrist R', alpha=0.5)
    plt.plot(ts, v_il, label='Index L', alpha=0.5)
    plt.plot(ts, v_ir, label='Index R', alpha=0.5)
    plt.plot(ts, v_avg, label='Avg Vel', color='black', linewidth=2)
    
    plt.axhline(0.012, color='green', linestyle='--', label='START_THRESH (0.012)')
    plt.axhline(0.006, color='red', linestyle='--', label='STOP_THRESH (0.006)')
    
    plt.xlim(3.0, max(ts))
    plt.title('Velocity Curve for 06_gestures_short.mp4')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Velocity (stride=3)')
    plt.legend()
    plt.grid(True)
    plt.savefig('data/velocity_plot.png')
    print("Saved data/velocity_plot.png")

if __name__ == '__main__':
    main()
