/**
 * ISL Recognition Pipeline — Feature Preprocessing
 *
 * This module reproduces the EXACT preprocessing from the training notebook
 * ISL_Sign_Recognition_Training_Final.ipynb.
 *
 * Verified training code (Cell 7 of training notebook):
 *
 * def normalize_sequence(seq):
 *     seq = seq.copy()
 *     pose = seq[:, :132].reshape(-1, 33, 4)
 *     lh   = seq[:, 132:195].reshape(-1, 21, 3)
 *     rh   = seq[:, 195:258].reshape(-1, 21, 3)
 *
 *     left_sh, right_sh = pose[:, 11, :2], pose[:, 12, :2]
 *     center = (left_sh + right_sh) / 2.0
 *     scale  = np.linalg.norm(left_sh - right_sh, axis=1, keepdims=True)
 *     scale  = np.where(scale < 1e-4, 1.0, scale)
 *
 *     pose[:, :, :2] -= center[:, None, :]
 *     pose[:, :, :2] /= scale[:, None, :]
 *     lh[:, :, :2]   -= center[:, None, :]
 *     lh[:, :, :2]   /= scale[:, None, :]
 *     rh[:, :, :2]   -= center[:, None, :]
 *     rh[:, :, :2]   /= scale[:, None, :]
 *
 *     return np.concatenate([
 *         pose.reshape(-1, 132), lh.reshape(-1, 63), rh.reshape(-1, 63)
 *     ], axis=1)
 *
 * Feature layout per frame (258 values total):
 *   [0..131]   33 pose landmarks × (x, y, z, visibility)
 *   [132..194] 21 left-hand landmarks × (x, y, z)
 *   [195..257] 21 right-hand landmarks × (x, y, z)
 *
 * Normalization per frame:
 *   - Left shoulder  = pose landmark index 11 → x,y = frame[11*4], frame[11*4+1]
 *   - Right shoulder = pose landmark index 12 → x,y = frame[12*4], frame[12*4+1]
 *   - center  = midpoint of shoulder pair (x,y only)
 *   - scale   = Euclidean distance between shoulders (x,y only)
 *   - if scale < 1e-4 → scale = 1.0
 *   - Subtract center from every landmark's x,y
 *   - Divide every landmark's x,y by scale
 *   - z coordinates: UNCHANGED
 *   - visibility values: UNCHANGED
 */

import type { FrameLandmarks } from './types'

/** Total features per frame: 33*4 + 21*3 + 21*3 = 258 */
export const FEATURE_DIM = 258

/** Pose segment: 33 landmarks × 4 channels = 132 values */
const POSE_DIM = 132
/** Hand segment: 21 landmarks × 3 channels = 63 values */
const HAND_DIM = 63

/** Pose landmark 11 = left shoulder */
const LEFT_SHOULDER_IDX = 11
/** Pose landmark 12 = right shoulder */
const RIGHT_SHOULDER_IDX = 12

/**
 * Constructs a raw (un-normalized) 258-dimensional feature vector from
 * one video frame's landmark detections.
 *
 * Missing landmarks produce zero-filled segments (exactly as training).
 * Always returns Float32Array(258). Never throws for missing landmarks.
 */
export function extractFeatureVector(landmarks: FrameLandmarks): Float32Array {
  const features = new Float32Array(FEATURE_DIM)

  // ── Pose: 33 × (x, y, z, visibility) = 132 ───────────────────────
  if (landmarks.pose !== null) {
    if (landmarks.pose.length !== POSE_DIM) {
      console.warn(`[ISL] Pose array has wrong length ${landmarks.pose.length}, expected ${POSE_DIM}`)
    } else {
      features.set(landmarks.pose, 0)
    }
  }
  // else: pose segment remains zeros (correct training behavior)

  // ── Left hand: 21 × (x, y, z) = 63 ───────────────────────────────
  if (landmarks.leftHand !== null) {
    if (landmarks.leftHand.length !== HAND_DIM) {
      console.warn(`[ISL] Left hand array has wrong length ${landmarks.leftHand.length}, expected ${HAND_DIM}`)
    } else {
      features.set(landmarks.leftHand, POSE_DIM)
    }
  }
  // else: left hand segment remains zeros

  // ── Right hand: 21 × (x, y, z) = 63 ─────────────────────────────
  if (landmarks.rightHand !== null) {
    if (landmarks.rightHand.length !== HAND_DIM) {
      console.warn(`[ISL] Right hand array has wrong length ${landmarks.rightHand.length}, expected ${HAND_DIM}`)
    } else {
      features.set(landmarks.rightHand, POSE_DIM + HAND_DIM)
    }
  }
  // else: right hand segment remains zeros

  // Invariant: features.length is always FEATURE_DIM (Float32Array is fixed size)
  if (features.length !== FEATURE_DIM) {
    throw new Error(
      `[ISL] extractFeatureVector: produced ${features.length} features, expected ${FEATURE_DIM}. ` +
      'This is a bug in the feature construction code.'
    )
  }

  return features
}

/**
 * Applies the EXACT training normalization to a window of raw feature frames.
 *
 * Implements normalize_sequence() from the training notebook verbatim.
 * Applied per-frame (each frame gets its own shoulder center/scale).
 *
 * Input:  rawFrames — array of raw Float32Array(258) frames
 * Output: new Float32Array of length (frames × 258) with normalization applied
 *
 * Does NOT resample. The caller must supply exactly WINDOW_SIZE frames.
 */
export function normalizeWindow(rawFrames: Float32Array[]): Float32Array[] {
  const n = rawFrames.length
  const result: Float32Array[] = new Array(n)

  for (let f = 0; f < n; f++) {
    const frame = rawFrames[f]
    // Work on a copy to avoid mutating the rolling buffer
    const norm = new Float32Array(frame)

    // ── Read pose as 33 × 4 ────────────────────────────────────────────
    // Pose starts at index 0. Landmark i: frame[i*4], frame[i*4+1], frame[i*4+2], frame[i*4+3]

    // Left shoulder (index 11): x = frame[11*4], y = frame[11*4+1]
    const lsBase = LEFT_SHOULDER_IDX * 4
    const lsx = frame[lsBase]
    const lsy = frame[lsBase + 1]

    // Right shoulder (index 12): x = frame[12*4], y = frame[12*4+1]
    const rsBase = RIGHT_SHOULDER_IDX * 4
    const rsx = frame[rsBase]
    const rsy = frame[rsBase + 1]

    // center = (left_shoulder + right_shoulder) / 2
    const cx = (lsx + rsx) / 2.0
    const cy = (lsy + rsy) / 2.0

    // scale = ||left_shoulder - right_shoulder||
    const dx = lsx - rsx
    const dy = lsy - rsy
    let scale = Math.sqrt(dx * dx + dy * dy)
    if (scale < 1e-4) scale = 1.0

    // ── Normalize pose x/y (33 landmarks × 4 channels) ────────────────
    for (let i = 0; i < 33; i++) {
      const base = i * 4
      norm[base]     = (frame[base]     - cx) / scale  // x
      norm[base + 1] = (frame[base + 1] - cy) / scale  // y
      // norm[base + 2] = frame[base + 2]  (z — UNCHANGED)
      // norm[base + 3] = frame[base + 3]  (visibility — UNCHANGED)
    }

    // ── Normalize left hand x/y (21 landmarks × 3 channels) ───────────
    const lhStart = POSE_DIM // 132
    for (let i = 0; i < 21; i++) {
      const base = lhStart + i * 3
      norm[base]     = (frame[base]     - cx) / scale  // x
      norm[base + 1] = (frame[base + 1] - cy) / scale  // y
      // norm[base + 2] = frame[base + 2]  (z — UNCHANGED)
    }

    // ── Normalize right hand x/y (21 landmarks × 3 channels) ──────────
    const rhStart = POSE_DIM + HAND_DIM // 195
    for (let i = 0; i < 21; i++) {
      const base = rhStart + i * 3
      norm[base]     = (frame[base]     - cx) / scale  // x
      norm[base + 1] = (frame[base + 1] - cy) / scale  // y
      // norm[base + 2] = frame[base + 2]  (z — UNCHANGED)
    }

    // Assertion: output frame must still be 258 values
    if (norm.length !== FEATURE_DIM) {
      throw new Error(`[ISL] normalizeWindow: frame ${f} has ${norm.length} values after normalization, expected ${FEATURE_DIM}`)
    }

    result[f] = norm
  }

  return result
}
