import type { ISLPrediction } from './types'
import { getMvpModelState, runMvpInference } from './mvp_model'
import { getMvpLabelForIndex } from './mvp_labels'

export const MVP_CONFIDENCE_THRESHOLD = 0.5
export const MVP_WINDOW_SIZE = 40
const STRIDE = 2
export const MVP_SMOOTHING_WINDOW = 5

let frameBuffer: Float32Array[] = []
let predictionHistory: string[] = []

export function getMvpBufferFill(): number {
  return frameBuffer.length / MVP_WINDOW_SIZE
}

export function resetMvpPredictor(): void {
  frameBuffer = []
  predictionHistory = []
}

export async function addMvpFrame(normalizedFeatures: Float32Array): Promise<ISLPrediction | null> {
  // Prevent all-zero frames (no hands, no pose) from polluting the temporal buffer.
  let isAllZero = true
  for (let i = 0; i < normalizedFeatures.length; i++) {
    if (normalizedFeatures[i] !== 0) {
      isAllZero = false
      break
    }
  }

  if (isAllZero) {
    if (frameBuffer.length > 0) {
      console.log('[ISL MVP] Frame is all zeros — clearing buffer')
      frameBuffer = []
      predictionHistory = []
    }
    return null
  }

  frameBuffer.push(normalizedFeatures)

  if (frameBuffer.length < MVP_WINDOW_SIZE) {
    return null
  }

  if (frameBuffer.length > MVP_WINDOW_SIZE) {
    frameBuffer.shift()
  }

  const shouldPredict = (frameBuffer.length === MVP_WINDOW_SIZE) && (performance.now() % STRIDE === 0 || true) 
  // Let's just predict on every frame once buffer is full for MVP, or stride it.
  
  if (getMvpModelState() !== 'ready') {
    return null
  }

  try {
    const probs = await runMvpInference(frameBuffer)

    let maxProb = 0
    let maxIdx = -1

    for (let i = 0; i < probs.length; i++) {
      if (probs[i] > maxProb) {
        maxProb = probs[i]
        maxIdx = i
      }
    }

    let rawLabel = 'UNKNOWN'
    if (maxProb >= MVP_CONFIDENCE_THRESHOLD && maxIdx !== -1) {
      rawLabel = getMvpLabelForIndex(maxIdx)
    }

    // Temporal smoothing
    predictionHistory.push(rawLabel)
    if (predictionHistory.length > MVP_SMOOTHING_WINDOW) {
      predictionHistory.shift()
    }

    // Find most frequent label in history
    const counts: Record<string, number> = {}
    let maxCount = 0
    let smoothedLabel = 'Uncertain sign'

    for (const lbl of predictionHistory) {
      counts[lbl] = (counts[lbl] || 0) + 1
      if (counts[lbl] > maxCount) {
        maxCount = counts[lbl]
        smoothedLabel = lbl
      }
    }

    // If max count isn't strong enough (e.g., < 3 out of 5), remain uncertain
    if (maxCount < 3) {
      smoothedLabel = 'Uncertain sign'
    }

    console.log(`[ISL MVP] predicted index: ${maxIdx}`)
    console.log(`[ISL MVP] predicted label: ${rawLabel}`)
    console.log(`[ISL MVP] confidence: ${maxProb}`)

    return {
      label: smoothedLabel,
      confidence: maxProb,
      index: maxIdx,
    }
  } catch (err) {
    console.error('[ISL MVP] Inference error:', err)
    return null
  }
}
