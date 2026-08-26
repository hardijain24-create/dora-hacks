/**
 * ISL Recognition Pipeline — Predictor
 *
 * Manages the 40-frame rolling window, normalization, and inference.
 *
 * Rolling window strategy:
 *   - Append raw feature frames to a circular buffer.
 *   - When the buffer reaches WINDOW_SIZE (40 frames), inference is eligible.
 *   - A stride counter limits how often inference runs (every INFERENCE_STRIDE frames).
 *   - After inference: the buffer is NOT cleared. New frames slide the window.
 *
 * This matches the training: sequences of exactly 40 frames, normalized and inferred.
 * No temporal resampling is applied to live 40-frame windows.
 *
 * Confidence threshold: if top softmax probability < CONFIDENCE_THRESHOLD,
 * the result is returned with a 'low_confidence' flag but NOT suppressed —
 * the UI decides how to display it.
 *
 * All-zero frame guard:
 *   If too few frames in the rolling window have pose data (no person detected),
 *   inference is skipped to prevent nonsensical predictions from empty frames.
 */

import type { ISLPrediction } from './types'
import { normalizeWindow, FEATURE_DIM } from './preprocessing'
import { runInference } from './model'
import { getLabels, cleanLabel } from './labels'

/** Must match training: SEQ_LEN = 40 */
export const WINDOW_SIZE = 40

/**
 * Inference runs every INFERENCE_STRIDE new frames.
 * Lower = more frequent predictions but higher CPU load.
 * 5 is a good default for ~30fps camera (≈6 inferences/second).
 */
export const INFERENCE_STRIDE = 5

/** Minimum softmax confidence to consider a prediction confident. */
export const CONFIDENCE_THRESHOLD = 0.55

/**
 * Minimum number of frames in the rolling window that must have non-zero pose
 * before inference is attempted. Prevents inference on empty/no-person sequences.
 */
/** Throttle debug logs: emit at most once per this interval (ms) */
const DEBUG_LOG_INTERVAL_MS = 1500

/** Smoothing: last N predictions for dominant-class voting */
const SMOOTHING_WINDOW = 4

interface SmoothingEntry {
  index: number
  confidence: number
}

interface PredictorState {
  /** Raw feature frames in the rolling window */
  buffer: Float32Array[]
  /** Count of frames added since last inference */
  framesSinceInference: number
  /** Is inference currently running (prevents overlapping calls) */
  inferring: boolean
}

let _state: PredictorState = {
  buffer: [],
  framesSinceInference: 0,
  inferring: false,
}

let _smoothingHistory: SmoothingEntry[] = []
let _lastDebugLogMs = 0

/**
 * Returns true if the given raw frame has non-zero pose data.
 * Pose occupies indices 0..131 (33 × 4). If the first 8 values are all zero,
 * the frame has no detected person (pose was missing → zero-filled).
 */
/**
 * Resets the predictor state (rolling buffer, stride counter, smoothing history).
 * Call when starting a new live translation session.
 */
export function resetPredictor(): void {
  _state = {
    buffer: [],
    framesSinceInference: 0,
    inferring: false,
  }
  _smoothingHistory = []
  _lastDebugLogMs = 0
}

/**
 * Adds one raw feature frame to the rolling buffer.
 * Returns an ISLPrediction if inference ran during this call, or null.
 *
 * @param frame — raw Float32Array(258) from extractFeatureVector()
 * @returns prediction or null
 */
export async function addFrame(frame: Float32Array): Promise<ISLPrediction | null> {
  if (frame.length !== FEATURE_DIM) {
    console.warn(`[ISL] addFrame: frame has ${frame.length} features, expected ${FEATURE_DIM}`)
    return null
  }

  // Add to rolling buffer
  _state.buffer.push(frame)

  // Keep only the last WINDOW_SIZE frames (rolling window)
  if (_state.buffer.length > WINDOW_SIZE) {
    _state.buffer.shift()
  }

  _state.framesSinceInference++

  // Not enough frames yet
  if (_state.buffer.length < WINDOW_SIZE) {
    if ((performance.now() - _lastDebugLogMs) > DEBUG_LOG_INTERVAL_MS) {
        console.log(`[ISL] buffer: ${_state.buffer.length}/${WINDOW_SIZE}`)
        _lastDebugLogMs = performance.now()
    }
    return null
  }

  // Stride check: only infer every INFERENCE_STRIDE frames
  if (_state.framesSinceInference < INFERENCE_STRIDE) {
    if ((performance.now() - _lastDebugLogMs) > DEBUG_LOG_INTERVAL_MS) {
        console.log(`[ISL] buffer: ${_state.buffer.length}/${WINDOW_SIZE}`)
        _lastDebugLogMs = performance.now()
    }
    return null
  }

  // Already running inference (async gap protection)
  if (_state.inferring) {
    return null
  }

  _state.framesSinceInference = 0
  _state.inferring = true

  try {
    // Copy the current 40 raw frames
    const rawWindow = _state.buffer.slice(-WINDOW_SIZE)

    if (rawWindow.length !== WINDOW_SIZE) {
      return null
    }

    console.assert(
      rawWindow.every(f => f.length === FEATURE_DIM),
      '[ISL] Some frames in the window have wrong length'
    )

    // ── All-zero guard: skip inference when no person is in frame ─────────
    // If fewer than MIN_POSE_FRAMES frames have pose data (non-zero pose block),
    // the window is mostly empty → skip inference to avoid bogus predictions.
    // ─────────────────────────────────────────────────────────────────────

    // Throttled debug flag: emit diagnostic logs at most once per 1500ms
    const nowMs = performance.now()
    const shouldDebug = (nowMs - _lastDebugLogMs) > DEBUG_LOG_INTERVAL_MS
    if (shouldDebug) _lastDebugLogMs = nowMs

    // Apply EXACT training normalization (shoulder-centered x/y)
    const normalizedFrames = normalizeWindow(rawWindow)

    if (shouldDebug) {
      console.log(`[ISL] inference started`)
      console.log(`[ISL] input shape: [1,${WINDOW_SIZE},${FEATURE_DIM}]`)
    }

    // Run TensorFlow.js inference
    const probabilities = await runInference(normalizedFrames)

    if (probabilities.length !== 261) {
      throw new Error(`[ISL] Model output has ${probabilities.length} values, expected 261`)
    }

    // Argmax
    let maxIdx = 0
    let maxVal = probabilities[0]
    for (let i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > maxVal) {
        maxVal = probabilities[i]
        maxIdx = i
      }
    }

    if (maxIdx < 0 || maxIdx >= 261) {
      throw new Error(`[ISL] argmax index ${maxIdx} is out of range`)
    }

    // Get labels for raw prediction logging
    const labels = getLabels()
    if (labels === null) {
      console.warn('[ISL] Labels not loaded yet')
      return null
    }

    const rawLabelFull = labels[maxIdx]
    const rawDisplayLabel = cleanLabel(rawLabelFull)

    if (shouldDebug) {
      console.log(`[ISL] output shape: [1,${probabilities.length}]`)

      // Top 5
      const top5 = Array.from(probabilities)
        .map((p, i) => ({ p, i }))
        .sort((a, b) => b.p - a.p)
        .slice(0, 5)

      console.log(`[ISL] top-5 indices: [${top5.map(x => x.i).join(', ')}]`)
      console.log(`[ISL] top-5 probabilities: [${top5.map(x => x.p.toFixed(3)).join(', ')}]`)

      console.log(`[ISL] predicted index: ${maxIdx}`)
      console.log(`[ISL] predicted label: ${rawDisplayLabel}`)
      console.log(`[ISL] confidence: ${maxVal.toFixed(2)}`)
    }

    // Temporal smoothing: track recent predictions
    _smoothingHistory.push({ index: maxIdx, confidence: maxVal })
    if (_smoothingHistory.length > SMOOTHING_WINDOW) {
      _smoothingHistory.shift()
    }

    // Vote for dominant class in smoothing window
    const votes = new Map<number, number>()
    for (const entry of _smoothingHistory) {
      // suppress extremely low-confidence predictions
      if (entry.confidence < CONFIDENCE_THRESHOLD) {
        votes.set(-1, (votes.get(-1) ?? 0) + 1)
      } else {
        votes.set(entry.index, (votes.get(entry.index) ?? 0) + 1)
      }
    }
    
    let dominantIdx = -1
    let dominantVotes = 0
    for (const [idx, count] of votes) {
      if (count > dominantVotes) {
        dominantVotes = count
        dominantIdx = idx
      }
    }

    // require multiple consistent predictions before changing the displayed gesture
    if (dominantVotes < 3) {
      dominantIdx = -1
    }

    let displayLabel = 'Uncertain / No sign detected'
    let finalConfidence = 0

    if (dominantIdx !== -1) {
      const rawLabel = labels[dominantIdx]
      displayLabel = cleanLabel(rawLabel)
      
      let confSum = 0; let confCount = 0
      for (const entry of _smoothingHistory) {
        if (entry.index === dominantIdx) {
          confSum += entry.confidence
          confCount++
        }
      }
      finalConfidence = confCount > 0 ? confSum / confCount : 0
    }

    const prediction: ISLPrediction = {
      index: dominantIdx !== -1 ? dominantIdx : maxIdx,
      label: displayLabel,
      confidence: dominantIdx !== -1 ? finalConfidence : maxVal,
      // signature intentionally omitted — gesture signatures are display-only,
      // not part of the inference path
    }

    return prediction
  } catch (err) {
    console.log('[ISL] Inference error: ' + (err instanceof Error ? err.message : String(err)))
    return null
  } finally {
    _state.inferring = false
  }
}

/** Current buffer fill level as a fraction [0, 1]. */
export function getBufferFill(): number {
  return Math.min(_state.buffer.length / WINDOW_SIZE, 1)
}
