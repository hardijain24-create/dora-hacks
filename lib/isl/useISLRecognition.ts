'use client'

/**
 * ISL Recognition Pipeline — React Hook
 *
 * useISLRecognition integrates the entire ML pipeline into the existing React app:
 *   - Loads the TensorFlow.js model (once)
 *   - Loads labels (once)
 *   - Initializes MediaPipe detectors (once)
 *   - Runs the frame processing loop when isRunning === true
 *   - Returns prediction results and pipeline state
 *
 * This hook REUSES the existing video element via videoRef.
 * It does NOT create a second camera stream.
 *
 * Browser-only: all ML code runs after mount via useEffect.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import type { ISLPrediction, ModelState, LandmarkState } from '@/lib/isl/types'
import { CONFIDENCE_THRESHOLD, WINDOW_SIZE } from '@/lib/isl/predictor'
import { loadModel } from '@/lib/isl/model'
import { loadLabels } from '@/lib/isl/labels'
import * as landmarks from '@/lib/isl/landmarks'
import * as preprocessing from '@/lib/isl/preprocessing'
import * as predictor from '@/lib/isl/predictor'
import { loadMvpModel, getMvpModelState } from '@/lib/isl/mvp_model'
import { loadMvpLabels } from '@/lib/isl/mvp_labels'
import * as mvpPredictor from '@/lib/isl/mvp_predictor'

// ── Module-level state for singleton ML resources ────────────────────────────
// These are kept outside React state to prevent re-initialization on re-renders.
// They are accessed by the hook functions below.

interface ISLRecognitionState {
  prediction: ISLPrediction | null
  mvpPrediction: ISLPrediction | null
  modelState: ModelState
  mvpModelState: ModelState
  landmarkState: LandmarkState
  /** Buffer fill fraction [0,1] */
  bufferFill: number
  /** Whether a valid pose was seen in the last frame */
  hasPose: boolean
  /** Whether at least one hand was seen in the last frame */
  hasHand: boolean
}

export interface UseISLRecognitionReturn {
  prediction: ISLPrediction | null
  mvpPrediction: ISLPrediction | null
  modelState: ModelState
  mvpModelState: ModelState
  landmarkState: LandmarkState
  bufferFill: number
  hasPose: boolean
  hasHand: boolean
  /** Start the live recognition loop */
  startRecognition: () => void
  /** Stop the live recognition loop */
  stopRecognition: () => void
}

export function useISLRecognition(
  isRunning: boolean,
  videoRef: React.RefObject<HTMLVideoElement | null>
): UseISLRecognitionReturn {
  const [state, setState] = useState<ISLRecognitionState>({
    prediction: null,
    mvpPrediction: null,
    modelState: 'idle',
    mvpModelState: 'idle',
    landmarkState: 'idle',
    bufferFill: 0,
    hasPose: false,
    hasHand: false,
  })

  // Refs that survive re-renders without causing them
  const rafIdRef = useRef<number | null>(null)
  const isLoopRunningRef = useRef(false)
  const lastTimestampRef = useRef(0)
  const framesSinceFirstRef = useRef(false)
  /** Throttle debug logs — only emit ~once per second */
  const lastDebugLogRef = useRef(0)

  // ── Load model and labels on mount ──────────────────────────────────────
  useEffect(() => {
    if (typeof window === 'undefined') return

    let cancelled = false

    async function initML() {
      setState(prev => ({ ...prev, modelState: 'loading' }))

      try {
        await Promise.all([loadModel(), loadLabels()])
        // MVP model loading is best-effort since it might not be trained yet
        Promise.all([loadMvpModel(), loadMvpLabels()]).then(() => {
          if (!cancelled) setState(prev => ({ ...prev, mvpModelState: 'ready' }))
        }).catch(() => {
          if (!cancelled) setState(prev => ({ ...prev, mvpModelState: 'error' }))
        })
        if (!cancelled) {
          console.log('[ISL] Model ready')
          setState(prev => ({ ...prev, modelState: 'ready' }))
        }
      } catch (err) {
        console.error('[ISL] ML initialization error:', err)
        if (!cancelled) {
          setState(prev => ({ ...prev, modelState: 'error' }))
        }
      }
    }

    initML()
    return () => { cancelled = true }
  }, [])

  // ── Frame processing loop ────────────────────────────────────────────────
  const startLoop = useCallback(() => {
    if (isLoopRunningRef.current) return
    isLoopRunningRef.current = true
    lastTimestampRef.current = 0

    // Static imports are now used instead of loadModules()

    async function processFrame() {
      if (!isLoopRunningRef.current) return

      // Pause when tab is hidden
      if (document.visibilityState === 'hidden') {
        rafIdRef.current = requestAnimationFrame(() => { void processFrame() })
        return
      }

      const video = videoRef.current
      if (!video || video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0 || video.paused || video.ended) {
        rafIdRef.current = requestAnimationFrame(() => { void processFrame() })
        return
      }

      if (landmarks === null || preprocessing === null || predictor === null) {
        rafIdRef.current = requestAnimationFrame(() => { void processFrame() })
        return
      }

      // Skip frames until MediaPipe is fully ready
      if (landmarks.getLandmarkState() !== 'ready') {
        rafIdRef.current = requestAnimationFrame(() => { void processFrame() })
        return
      }

      // Monotonically increasing timestamp required by MediaPipe VIDEO mode
      const now = performance.now()
      const ts = now > lastTimestampRef.current ? now : lastTimestampRef.current + 1
      lastTimestampRef.current = ts

      try {
        if (!framesSinceFirstRef.current) {
          framesSinceFirstRef.current = true
          console.log('[ISL] camera ready')
        }

        const debugNow = performance.now()
        const shouldDebug = debugNow - lastDebugLogRef.current > 1000

        // Detect landmarks
        const lmResult = landmarks.detectLandmarks(video, ts)

        if (shouldDebug) {
          lastDebugLogRef.current = debugNow
          if (lmResult.pose || lmResult.leftHand || lmResult.rightHand) {
            console.log('[ISL] landmarks: OK')
          }
        }

        // Build 258-feature vector
        const frame = preprocessing.extractFeatureVector(lmResult)

        // Validate feature vector length — required diagnostic
        if (frame.length !== 258) {
          console.error(`[ISL] Invalid feature vector length: ${frame.length}`)
        } else if (shouldDebug) {
          console.log(`[ISL] feature length: ${frame.length}`)
        }

        // Run prediction (may return null if buffer not full or stride not reached)
        const prediction = await predictor.addFrame(frame)
        const mvpPrediction = await mvpPredictor.addMvpFrame(frame)
        
        // predictor internally logs [ISL] FRAME BUFFER, TENSOR, and PREDICT CALLED.
        
        // Update React state only when something meaningful changes
        const fill = predictor.getBufferFill()
        const hasPose = lmResult.pose !== null
        const hasHand = lmResult.leftHand !== null || lmResult.rightHand !== null

        setState(prev => {
          const predChanged = prediction !== null && (
            prediction.label !== prev.prediction?.label ||
            Math.abs(prediction.confidence - (prev.prediction?.confidence ?? 0)) > 0.02
          )
          const fillChanged = Math.abs(fill - prev.bufferFill) > 0.05
          const poseChanged = hasPose !== prev.hasPose
          const handChanged = hasHand !== prev.hasHand

          if (!predChanged && !fillChanged && !poseChanged && !handChanged && mvpPrediction === prev.mvpPrediction) {
            return prev
          }

          return {
            ...prev,
            prediction: prediction !== null ? prediction : prev.prediction,
            mvpPrediction: mvpPrediction !== null ? mvpPrediction : prev.mvpPrediction,
            bufferFill: fill,
            hasPose,
            hasHand,
          }
        })
      } catch (err) {
        console.error('[ISL] Frame processing error:', err)
      }

      if (isLoopRunningRef.current) {
        rafIdRef.current = requestAnimationFrame(() => { void processFrame() })
      }
    }

    rafIdRef.current = requestAnimationFrame(() => { void processFrame() })
  }, [videoRef])

  const stopLoop = useCallback(() => {
    isLoopRunningRef.current = false
    framesSinceFirstRef.current = false
    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current)
      rafIdRef.current = null
    }
    // Reset predictor buffer
    import('@/lib/isl/predictor').then(({ resetPredictor }) => resetPredictor()).catch(() => {})
    import('@/lib/isl/mvp_predictor').then(({ resetMvpPredictor }) => resetMvpPredictor()).catch(() => {})
  }, [])

  // ── Initialize MediaPipe when we start running ───────────────────────────
  useEffect(() => {
    if (!isRunning) {
      stopLoop()
      setState(prev => ({ ...prev, bufferFill: 0, hasPose: false, hasHand: false }))
      return
    }

    // Start MediaPipe + loop
    async function startML() {
      try {
        setState(prev => ({ ...prev, landmarkState: 'initializing' }))
        await landmarks.initLandmarkers()
        if (landmarks.getLandmarkState() === 'ready') {
          console.log('[ISL] vision ready')
          setState(prev => ({ ...prev, landmarkState: 'ready' }))
        }
        startLoop()
      } catch (err) {
        console.error('[ISL] Failed to start ML:', err)
        setState(prev => ({ ...prev, landmarkState: 'error' }))
      }
    }

    startML()
    return () => stopLoop()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isRunning])

  // ── Handle tab visibility ────────────────────────────────────────────────
  useEffect(() => {
    function onVisibilityChange() {
      if (document.visibilityState === 'visible' && isRunning && !isLoopRunningRef.current) {
        startLoop()
      }
    }
    document.addEventListener('visibilitychange', onVisibilityChange)
    return () => document.removeEventListener('visibilitychange', onVisibilityChange)
  }, [isRunning, startLoop])

  // ── Cleanup on unmount ───────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      stopLoop()
      import('@/lib/isl/landmarks').then(({ closeLandmarkers }) => closeLandmarkers()).catch(() => {})
    }
  }, [stopLoop])

  const startRecognition = useCallback(() => {
    if (!isLoopRunningRef.current) startLoop()
  }, [startLoop])

  const stopRecognition = useCallback(() => {
    stopLoop()
  }, [stopLoop])

  return {
    prediction: state.prediction,
    mvpPrediction: state.mvpPrediction,
    modelState: state.modelState,
    mvpModelState: state.mvpModelState,
    landmarkState: state.landmarkState,
    bufferFill: state.bufferFill,
    hasPose: state.hasPose,
    hasHand: state.hasHand,
    startRecognition,
    stopRecognition,
  }
}

// Re-export constants for use in page.tsx
export { CONFIDENCE_THRESHOLD, WINDOW_SIZE }
