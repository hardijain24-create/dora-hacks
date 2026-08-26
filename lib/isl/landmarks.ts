/**
 * ISL Recognition Pipeline — MediaPipe Landmark Detector Service
 *
 * Initializes PoseLandmarker and HandLandmarker once.
 * Processes the existing live video element per frame.
 * Returns raw landmark data in the exact 258-feature layout.
 *
 * Training used MediaPipe Tasks Vision PoseLandmarker + HandLandmarker.
 * The browser must use the same detectors for landmark parity.
 *
 * IMPORTANT: Handedness is determined by MediaPipe's category label
 * ("Left" / "Right"), NOT by result array index.
 */

import type { FrameLandmarks } from './types'
import type { LandmarkState } from './types'

// MediaPipe Tasks Vision types (imported lazily to avoid SSR)
type MPVision = typeof import('@mediapipe/tasks-vision')
type PoseLandmarker = import('@mediapipe/tasks-vision').PoseLandmarker
type HandLandmarker = import('@mediapipe/tasks-vision').HandLandmarker

/**
 * Local WASM base path — served directly by Next.js from public/mediapipe/wasm
 * Using local WASM avoids CDN version mismatches and external network dependency.
 */
const MP_WASM_URL = '/mediapipe/wasm'

const POSE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'

const HAND_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

let _mpVision: MPVision | null = null
let _poseLandmarker: PoseLandmarker | null = null
let _handLandmarker: HandLandmarker | null = null
let _landmarkState: LandmarkState = 'idle'
let _initPromise: Promise<void> | null = null

export function getLandmarkState(): LandmarkState {
  return _landmarkState
}

/**
 * Initializes PoseLandmarker and HandLandmarker exactly once.
 * Safe to call multiple times — idempotent.
 */
export async function initLandmarkers(): Promise<void> {
  if (_landmarkState === 'ready') return
  if (_initPromise !== null) return _initPromise

  _landmarkState = 'initializing'
  console.log('[ISL] MediaPipe initializing...')

  _initPromise = (async () => {
    try {
      const vision = await import('@mediapipe/tasks-vision')
      _mpVision = vision

      const { FilesetResolver, PoseLandmarker: PL, HandLandmarker: HL } = vision

      const filesetResolver = await FilesetResolver.forVisionTasks(MP_WASM_URL)

      // Create PoseLandmarker in VIDEO mode
      _poseLandmarker = await PL.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath: POSE_MODEL_URL,
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      })

      // Create HandLandmarker in VIDEO mode
      _handLandmarker = await HL.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath: HAND_MODEL_URL,
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      })

      _landmarkState = 'ready'
      console.log('[ISL] POSE LANDMARKER READY')
      console.log('[ISL] HAND LANDMARKER READY')
      console.log('[ISL] MediaPipe ready')
    } catch (err) {
      _landmarkState = 'error'
      _initPromise = null
      throw new Error(`[ISL] MediaPipe initialization failed: ${String(err)}`)
    }
  })()

  return _initPromise
}

/**
 * Processes one video frame and returns raw landmark data.
 *
 * @param video — the existing live video element (already playing)
 * @param timestampMs — monotonically increasing timestamp in milliseconds
 * @returns FrameLandmarks with pose / leftHand / rightHand arrays (or null if not detected)
 */
export function detectLandmarks(
  video: HTMLVideoElement,
  timestampMs: number
): FrameLandmarks {
  const result: FrameLandmarks = {
    pose: null,
    leftHand: null,
    rightHand: null,
  }

  if (_poseLandmarker === null || _handLandmarker === null) {
    return result
  }

  // ── Pose detection ─────────────────────────────────────────────────
  try {
    const poseResult = _poseLandmarker.detectForVideo(video, timestampMs)
    if (poseResult.landmarks && poseResult.landmarks.length > 0) {
      const lm = poseResult.landmarks[0]
      const poseFeatures = new Float32Array(33 * 4)
      for (let i = 0; i < 33; i++) {
        const base = i * 4
        poseFeatures[base]     = lm[i].x
        poseFeatures[base + 1] = lm[i].y
        poseFeatures[base + 2] = lm[i].z
        poseFeatures[base + 3] = lm[i].visibility ?? 0
      }
      result.pose = poseFeatures
    }
  } catch (err) {
    // Pose failure → pose remains null (zero-filled in feature vector)
    console.warn('[ISL] Pose detection error:', err)
  }

  // ── Hand detection ─────────────────────────────────────────────────
  try {
    const handResult = _handLandmarker.detectForVideo(video, timestampMs)

    if (handResult.landmarks && handResult.landmarks.length > 0) {
      for (let h = 0; h < handResult.landmarks.length; h++) {
        const lm = handResult.landmarks[h]
        // Determine handedness from MediaPipe metadata — NOT array index
        // handResult.handedness[h][0].categoryName === 'Left' | 'Right'
        const handednessCategory = handResult.handedness?.[h]?.[0]?.categoryName ?? ''

        const handFeatures = new Float32Array(21 * 3)
        for (let i = 0; i < 21; i++) {
          const base = i * 3
          handFeatures[base]     = lm[i].x
          handFeatures[base + 1] = lm[i].y
          handFeatures[base + 2] = lm[i].z
        }

        // MediaPipe handedness from a mirrored (selfie) camera is inverted.
        // When video facingMode='user', MediaPipe 'Left' corresponds to the
        // user's RIGHT hand in the image, and vice versa.
        // The training used the same setup, so we replicate it exactly:
        // store MediaPipe's reported "Left" as leftHand and "Right" as rightHand.
        if (handednessCategory === 'Left') {
          result.leftHand = handFeatures
        } else if (handednessCategory === 'Right') {
          result.rightHand = handFeatures
        }
      }
    }
  } catch (err) {
    console.warn('[ISL] Hand detection error:', err)
  }

  return result
}

/**
 * Releases MediaPipe resources.
 * Call on component unmount.
 */
export async function closeLandmarkers(): Promise<void> {
  try {
    if (_poseLandmarker) {
      _poseLandmarker.close()
      _poseLandmarker = null
    }
    if (_handLandmarker) {
      _handLandmarker.close()
      _handLandmarker = null
    }
  } catch {
    // Ignore cleanup errors
  }
  _landmarkState = 'idle'
  _initPromise = null
}
