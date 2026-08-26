import type { ModelState } from './types'

type TF = typeof import('@tensorflow/tfjs')
type LayersModel = import('@tensorflow/tfjs').LayersModel
type Tensor = import('@tensorflow/tfjs').Tensor

/**
 * MVP model is served from a SEPARATE path from the INCLUDE model.
 * public/model/model.json  = original INCLUDE 261-class model (untouched)
 * public/models/isl-mvp/   = real MVP model (once trained + exported)
 */
const MODEL_URL = '/models/isl-mvp/model.json'
const LABELS_URL = '/models/isl-mvp/labels.json'
const EXPECTED_SEQ_LEN = 40
const EXPECTED_FEAT_DIM = 258

let _tf: TF | null = null
let _model: LayersModel | null = null
let _modelState: ModelState = 'idle'
let _loadPromise: Promise<LayersModel | null> | null = null
let _expectedClasses = -1

export function getMvpModelState(): ModelState {
  return _modelState
}

export function getMvpModel(): LayersModel | null {
  return _model
}

export async function loadMvpModel(): Promise<LayersModel | null> {
  if (_model !== null) return _model
  if (_loadPromise !== null) return _loadPromise

  _modelState = 'loading'
  console.log('[ISL MVP] Model loading...')

  _loadPromise = (async () => {
    try {
      const res = await fetch(LABELS_URL)
      if (res.ok) {
        const labels = await res.json()
        _expectedClasses = labels.length
        console.log(`[ISL MVP] labels loaded: ${_expectedClasses} classes`)
      } else {
        console.warn(`[ISL MVP] labels not found at ${LABELS_URL} — MVP model not yet trained.`)
        _modelState = 'error'
        return null
      }
    } catch (e) {
      console.warn('[ISL MVP] Label fetch failed', e)
      _modelState = 'error'
      return null
    }

    const tf = await import('@tensorflow/tfjs')
    await tf.setBackend('cpu')
    await tf.ready()
    _tf = tf

    let model: LayersModel
    try {
      model = await tf.loadLayersModel(MODEL_URL)
    } catch (err) {
      console.warn(`[ISL MVP] Failed to load model from ${MODEL_URL}: ${String(err)}`)
      _modelState = 'error'
      _loadPromise = null
      return null
    }

    const inputShape = model.inputs[0]?.shape
    if (
      inputShape == null ||
      inputShape.length !== 3 ||
      (inputShape[1] !== null && inputShape[1] !== EXPECTED_SEQ_LEN) ||
      (inputShape[2] !== null && inputShape[2] !== EXPECTED_FEAT_DIM)
    ) {
      console.warn(`[ISL MVP] Unexpected model input shape: ${JSON.stringify(inputShape)}`)
      _modelState = 'error'
      return null
    }

    const outputShape = model.outputs[0]?.shape
    if (
      outputShape == null ||
      outputShape.length !== 2 ||
      (outputShape[1] !== null && outputShape[1] !== _expectedClasses)
    ) {
      console.warn(`[ISL MVP] Unexpected model output shape: ${JSON.stringify(outputShape)}`)
      _modelState = 'error'
      return null
    }

    console.log('[ISL MVP] model loaded')
    _model = model
    _modelState = 'ready'
    return model
  })()

  return _loadPromise
}

export async function runMvpInference(normalizedFrames: Float32Array[]): Promise<Float32Array> {
  if (_model === null) throw new Error('[ISL MVP] runMvpInference called before model is ready')
  if (_tf === null) throw new Error('[ISL MVP] TensorFlow.js not loaded')

  const tf = _tf
  const model = _model

  const seqLen = normalizedFrames.length
  const featDim = normalizedFrames[0]?.length ?? 0

  const flat = new Float32Array(seqLen * featDim)
  for (let i = 0; i < seqLen; i++) {
    flat.set(normalizedFrames[i], i * featDim)
  }

  let inputTensor: Tensor | null = null
  let outputTensor: Tensor | null = null

  try {
    inputTensor = tf.tensor3d(flat, [1, seqLen, featDim])
    const raw = model.predict(inputTensor)
    outputTensor = Array.isArray(raw) ? raw[0] : raw
    console.log(`[ISL MVP] inference output: [${outputTensor.shape.join(', ')}]`)
    const probs = new Float32Array(await outputTensor.data())

    if (probs.length !== _expectedClasses) {
      throw new Error(`[ISL MVP] Model output has ${probs.length} values, expected ${_expectedClasses}`)
    }

    return probs
  } catch (err) {
    console.log('[ISL MVP] Error in runMvpInference: ' + String(err))
    throw err
  } finally {
    inputTensor?.dispose()
    outputTensor?.dispose()
  }
}
