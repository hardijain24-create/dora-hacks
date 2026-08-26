/**
 * ISL Recognition Pipeline — Shared Types
 * All browser-only. Never imported during SSR.
 */

/** A single ISL recognition prediction from the TensorFlow.js model. */
export interface ISLPrediction {
  /** argmax index into the labels array */
  index: number
  /** Display-ready label (numeric prefix stripped for UI only) */
  label: string
  /** Raw softmax probability [0, 1] */
  confidence: number
  /** Diagnostic gesture signature data */
  signature?: any
}

/** Current state of the TensorFlow.js model loader. */
export type ModelState = 'idle' | 'loading' | 'ready' | 'error'

/** Current state of the MediaPipe landmark detectors. */
export type LandmarkState = 'idle' | 'initializing' | 'ready' | 'error'

/**
 * Raw per-frame landmark data extracted from one video frame.
 * All values are in normalized image coordinates [0,1].
 */
export interface FrameLandmarks {
  /**
   * 33 pose landmarks × (x, y, z, visibility) = 132 values.
   * null when pose is not detected (caller should zero-fill).
   */
  pose: Float32Array | null
  /**
   * 21 left-hand landmarks × (x, y, z) = 63 values.
   * null when the left hand is not detected (caller zero-fills).
   */
  leftHand: Float32Array | null
  /**
   * 21 right-hand landmarks × (x, y, z) = 63 values.
   * null when the right hand is not detected (caller zero-fills).
   */
  rightHand: Float32Array | null
}
