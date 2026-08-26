#!/usr/bin/env python3
"""
scripts/02_extract_features.py

Phase 2: Extract landmark features from labeled gesture segments.

Reads:  data/segment_manifest_labeled.json
        (produced by 01_segment_videos.py, with labels filled in by you)

Writes: data/mvp_features/X.npy        shape (N, 40, 258) float32
        data/mvp_features/y.npy        shape (N,)         int32 class indices
        data/mvp_features/labels.json  ["CLASS_A", "CLASS_B", ...]
        data/mvp_features/report.json  per-class sample counts + quality stats

Each output sample:
  - Contains exactly 40 frames × 258 features
  - Is shoulder-center normalized (matches preprocessing.ts exactly)
  - Is resampled via nearest-index from the raw captured segment

Usage:
  python scripts/02_extract_features.py
  python scripts/02_extract_features.py --manifest data/segment_manifest_labeled.json
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from collections import Counter

# ── Configuration ─────────────────────────────────────────────────────────────

POSE_MODEL_PATH = 'isl_tfjs_export (1)/pose_landmarker_lite.task'
HAND_MODEL_PATH = 'isl_tfjs_export (1)/hand_landmarker.task'

SEQ_LEN      = 40       # must match model architecture
FEATURE_DIM  = 258      # 33*4 + 21*3 + 21*3
MIN_SAMPLES  = 3        # warn if any class has fewer than this many samples

OUT_DIR = 'data/mvp_features'

# ── MediaPipe setup ───────────────────────────────────────────────────────────

BaseOptions       = mp_python.BaseOptions
VisionRunningMode = mp_vision.RunningMode

def _make_detectors():
    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
    )
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
    )
    return (
        mp_vision.PoseLandmarker.create_from_options(pose_opts),
        mp_vision.HandLandmarker.create_from_options(hand_opts),
    )

# ── Feature extraction (matches preprocessing.ts exactly) ────────────────────

def _extract_frame(frame_rgb, pose_det, hand_det):
    """
    Extract 258-feature vector from one frame.
    Layout: pose(33×4) + leftHand(21×3) + rightHand(21×3)
    Zero-fills missing landmarks.
    """
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    pose_r = pose_det.detect(mp_img)
    hand_r = hand_det.detect(mp_img)

    feat = np.zeros(FEATURE_DIM, dtype=np.float32)

    # Pose: 33 × (x, y, z, visibility) = 132 values
    if pose_r.pose_landmarks:
        lms = pose_r.pose_landmarks[0]
        for i, lm in enumerate(lms[:33]):
            base = i * 4
            feat[base]     = lm.x
            feat[base + 1] = lm.y
            feat[base + 2] = lm.z
            feat[base + 3] = getattr(lm, 'visibility', 0.0)

    # Hands: left=132..194, right=195..257 (each 21×3)
    if hand_r.hand_landmarks:
        for hand_lms, handedness in zip(hand_r.hand_landmarks, hand_r.handedness):
            label = handedness[0].category_name  # 'Left' or 'Right'
            offset = 132 if label == 'Left' else 195
            for i, lm in enumerate(hand_lms[:21]):
                base = offset + i * 3
                feat[base]     = lm.x
                feat[base + 1] = lm.y
                feat[base + 2] = lm.z

    return feat


def normalize_sequence(seq):
    """
    Exact mirror of normalizeWindow() in preprocessing.ts.
    Shoulder-centered, shoulder-width-scaled, per frame.
    """
    seq = seq.copy()
    pose  = seq[:, :132].reshape(-1, 33, 4)
    lh    = seq[:, 132:195].reshape(-1, 21, 3)
    rh    = seq[:, 195:258].reshape(-1, 21, 3)

    left_sh  = pose[:, 11, :2]   # landmark 11 = left shoulder
    right_sh = pose[:, 12, :2]   # landmark 12 = right shoulder
    center   = (left_sh + right_sh) / 2.0
    scale    = np.linalg.norm(left_sh - right_sh, axis=1, keepdims=True)
    scale    = np.where(scale < 1e-4, 1.0, scale)

    pose[:, :, :2] -= center[:, None, :]
    pose[:, :, :2] /= scale[:, None, :]
    lh[:, :, :2]   -= center[:, None, :]
    lh[:, :, :2]   /= scale[:, None, :]
    rh[:, :, :2]   -= center[:, None, :]
    rh[:, :, :2]   /= scale[:, None, :]

    return np.concatenate([
        pose.reshape(-1, 132),
        lh.reshape(-1, 63),
        rh.reshape(-1, 63),
    ], axis=1)


def resample_sequence(seq, target_len=SEQ_LEN):
    """
    Nearest-index resampling to exactly target_len frames.
    Matches resample_sequence() in build_mvp_dataset.py.
    """
    n = len(seq)
    if n == 0:
        return np.zeros((target_len, FEATURE_DIM), dtype=np.float32)
    if n == target_len:
        return seq
    idx = np.round(np.linspace(0, n - 1, target_len)).astype(int)
    return seq[idx]


def extract_segment(video_path, start_sec, end_sec, pose_det, hand_det):
    """
    Extract all frames from [start_sec, end_sec], apply normalization,
    resample to SEQ_LEN frames. Returns (N_raw, 40, 258) or None on error.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ERROR: cannot open {video_path}")
        return None, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec   * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    raw_frames = []
    for fi in range(start_frame, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        feat = _extract_frame(frame_rgb, pose_det, hand_det)
        raw_frames.append(feat)

    cap.release()

    n_raw = len(raw_frames)
    if n_raw == 0:
        return None, 0

    seq = np.array(raw_frames, dtype=np.float32)
    seq = normalize_sequence(seq)
    seq = resample_sequence(seq, SEQ_LEN)

    assert seq.shape == (SEQ_LEN, FEATURE_DIM), \
        f"Bad shape after resample: {seq.shape}"

    return seq, n_raw

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='data/segment_manifest_labeled.json')
    args = parser.parse_args()

    # Validate manifest
    if not os.path.exists(args.manifest):
        print(f"ERROR: manifest not found: {args.manifest}")
        print(f"Run scripts/01_segment_videos.py first, fill in labels,")
        print(f"then save as data/segment_manifest_labeled.json")
        sys.exit(1)

    with open(args.manifest) as f:
        manifest = json.load(f)

    # Filter out unlabeled or empty-label entries
    labeled = [e for e in manifest if e.get('label', '').strip()]
    skipped = len(manifest) - len(labeled)
    if skipped:
        print(f"INFO: Skipping {skipped} unlabeled segments")
    if not labeled:
        print("ERROR: No labeled segments found. Fill in 'label' fields first.")
        sys.exit(1)

    # Build sorted class list
    all_labels = sorted(set(e['label'] for e in labeled))
    label_to_idx = {l: i for i, l in enumerate(all_labels)}

    print(f"\nClasses ({len(all_labels)}):")
    counts = Counter(e['label'] for e in labeled)
    for cls in all_labels:
        n = counts[cls]
        warn = " [WARN: low]" if n < MIN_SAMPLES else ""
        print(f"  [{label_to_idx[cls]:2d}] {cls}: {n} sample(s){warn}")

    # Initialize MediaPipe
    print("\nInitializing MediaPipe...")
    pose_det, hand_det = _make_detectors()
    print("Ready.\n")

    X_list = []
    y_list = []
    report = []

    total = len(labeled)
    for i, entry in enumerate(labeled):
        video   = entry['video']
        start_s = entry['start_sec']
        end_s   = entry['end_sec']
        label   = entry['label']
        idx     = label_to_idx[label]

        print(f"[{i+1}/{total}] {label}  {video}  {start_s:.1f}–{end_s:.1f}s")

        if not os.path.exists(video):
            print(f"  SKIP: video file missing")
            report.append({'segment': entry, 'status': 'missing_video'})
            continue

        seq, n_raw = extract_segment(video, start_s, end_s, pose_det, hand_det)

        if seq is None:
            print(f"  SKIP: no frames extracted")
            report.append({'segment': entry, 'status': 'no_frames'})
            continue

        # Quality check: non-zero frame ratio
        nonzero = sum(1 for f in seq if f.any())
        ratio   = nonzero / SEQ_LEN

        status = 'ok' if ratio >= 0.5 else 'low_quality'
        if ratio < 0.5:
            print(f"  WARNING: only {nonzero}/{SEQ_LEN} non-zero frames ({ratio:.0%}) "
                  f"- pose/hands not detected in most frames")

        print(f"  raw={n_raw} frames -> resampled to {SEQ_LEN}, "
              f"non-zero={nonzero}/{SEQ_LEN} ({ratio:.0%})")

        X_list.append(seq)
        y_list.append(idx)
        report.append({
            'segment': entry,
            'status': status,
            'raw_frames': n_raw,
            'nonzero_frames': nonzero,
            'nonzero_ratio': round(ratio, 3),
        })

    if not X_list:
        print("\nERROR: No valid samples extracted.")
        sys.exit(1)

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, 40, 258)
    y = np.array(y_list, dtype=np.int32)               # (N,)

    print(f"\nFinal dataset: X={X.shape}, y={y.shape}")

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, 'X.npy'), X)
    np.save(os.path.join(OUT_DIR, 'y.npy'), y)

    with open(os.path.join(OUT_DIR, 'labels.json'), 'w') as f:
        json.dump(all_labels, f, indent=2)

    with open(os.path.join(OUT_DIR, 'report.json'), 'w') as f:
        json.dump({
            'total_samples': len(X_list),
            'classes': all_labels,
            'class_counts': {l: int(counts[l]) for l in all_labels},
            'samples': report,
        }, f, indent=2)

    print(f"\nSaved to {OUT_DIR}/")
    print(f"  X.npy:       {X.shape} float32")
    print(f"  y.npy:       {y.shape} int32")
    print(f"  labels.json: {len(all_labels)} classes")
    print(f"  report.json: extraction quality log")
    print(f"\nNEXT STEP: node scripts/03_train_mvp.mjs")


if __name__ == '__main__':
    main()
