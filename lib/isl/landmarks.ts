import type { LandmarkState } from './types'

type MPVision = typeof import('@mediapipe/tasks-vision')
type PoseLandmarker = import('@mediapipe/tasks-vision').PoseLandmarker
type HandLandmarker = import('@mediapipe/tasks-vision').HandLandmarker

/**
 * Local WASM base path — served directly by Next.js from public/mediapipe/wasm
 * Using local WASM avoids CDN version mismatches and external network dependency.
 */
const MP_WASM_URL = '/mediapipe/wasm'

const POSE_MODEL_URL = '/mediapipe/models/pose_landmarker_lite.task'
const HAND_MODEL_URL = '/mediapipe/models/hand_landmarker.task'

const CDN_POSE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'
const CDN_HAND_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'

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

      let filesetResolver
      try {
        filesetResolver = await FilesetResolver.forVisionTasks(MP_WASM_URL)
      } catch (err) {
        console.warn('[ISL] Local WASM resolver failed, falling back to CDN WASM:', err)
        filesetResolver = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm')
      }

      const createPose = async (modelPath: string, delegate: 'GPU' | 'CPU') => {
        return PL.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: modelPath,
            delegate,
          },
          runningMode: 'VIDEO',
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        })
      }

      const createHand = async (modelPath: string, delegate: 'GPU' | 'CPU') => {
        return HL.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: modelPath,
            delegate,
          },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.25,
          minHandPresenceConfidence: 0.25,
          minTrackingConfidence: 0.25,
        })
      }

      // Try local pose model first, fallback to CDN if 404
      try {
        try {
          _poseLandmarker = await createPose(POSE_MODEL_URL, 'GPU')
        } catch {
          _poseLandmarker = await createPose(POSE_MODEL_URL, 'CPU')
        }
      } catch (localErr) {
        console.warn('[ISL] Local PoseLandmarker model failed, falling back to CDN model:', localErr)
        try {
          _poseLandmarker = await createPose(CDN_POSE_MODEL_URL, 'GPU')
        } catch {
          _poseLandmarker = await createPose(CDN_POSE_MODEL_URL, 'CPU')
        }
      }

      // Try local hand model first, fallback to CDN if 404
      try {
        try {
          _handLandmarker = await createHand(HAND_MODEL_URL, 'GPU')
        } catch {
          _handLandmarker = await createHand(HAND_MODEL_URL, 'CPU')
        }
      } catch (localErr) {
        console.warn('[ISL] Local HandLandmarker model failed, falling back to CDN model:', localErr)
        try {
          _handLandmarker = await createHand(CDN_HAND_MODEL_URL, 'GPU')
        } catch {
          _handLandmarker = await createHand(CDN_HAND_MODEL_URL, 'CPU')
        }
      }

      _landmarkState = 'ready'
      console.log('[ISL] POSE LANDMARKER READY')
      console.log('[ISL] HAND LANDMARKER READY')
      console.log('[ISL] MediaPipe ready')
    } catch (err) {
      _landmarkState = 'error'
      _initPromise = null
      console.error('[ISL] MediaPipe init failed:', err)
    }
  })()

  return _initPromise
}

export function detectFrame(
  video: HTMLVideoElement,
  timestampMs: number
): {
  pose: Float32Array | null
  leftHand: Float32Array | null
  rightHand: Float32Array | null
} | null {
  if (_landmarkState !== 'ready' || !_poseLandmarker || !_handLandmarker) {
    return null
  }

  if (!video || video.readyState < 2 || video.paused || video.ended) {
    return null
  }

  try {
    const poseResult = _poseLandmarker.detectForVideo(video, timestampMs)
    const handResult = _handLandmarker.detectForVideo(video, timestampMs)

    let poseArr: Float32Array | null = null
    if (poseResult.landmarks && poseResult.landmarks.length > 0) {
      const lms = poseResult.landmarks[0]
      poseArr = new Float32Array(33 * 4)
      for (let i = 0; i < Math.min(lms.length, 33); i++) {
        poseArr[i * 4]     = lms[i].x
        poseArr[i * 4 + 1] = lms[i].y
        poseArr[i * 4 + 2] = lms[i].z
        poseArr[i * 4 + 3] = lms[i].visibility ?? 0.0
      }
    }

    let leftHandArr: Float32Array | null = null
    let rightHandArr: Float32Array | null = null

    if (
      handResult.landmarks &&
      handResult.landmarks.length > 0 &&
      handResult.handednesses &&
      handResult.handednesses.length > 0
    ) {
      for (let h = 0; h < handResult.landmarks.length; h++) {
        const lms = handResult.landmarks[h]
        const handedness = handResult.handednesses[h]
        if (!lms || !handedness || handedness.length === 0) continue

        const label = handedness[0].categoryName // 'Left' or 'Right'
        const arr = new Float32Array(21 * 3)
        for (let i = 0; i < Math.min(lms.length, 21); i++) {
          arr[i * 3]     = lms[i].x
          arr[i * 3 + 1] = lms[i].y
          arr[i * 3 + 2] = lms[i].z
        }

        if (label === 'Left') {
          leftHandArr = arr
        } else if (label === 'Right') {
          rightHandArr = arr
        }
      }
    }

    return { pose: poseArr, leftHand: leftHandArr, rightHand: rightHandArr }
  } catch (err) {
    console.warn('[ISL] detectFrame error:', err)
    return null
  }
}

export const detectLandmarks = detectFrame

export function closeLandmarkers(): void {
  if (_poseLandmarker) {
    try { _poseLandmarker.close() } catch {}
    _poseLandmarker = null
  }
  if (_handLandmarker) {
    try { _handLandmarker.close() } catch {}
    _handLandmarker = null
  }
  _landmarkState = 'idle'
  _initPromise = null
}

