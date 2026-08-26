/**
 * ISL Recognition Pipeline — TensorFlow.js Model Service
 *
 * Loads the model exactly once and caches it.
 * Browser-only module — never executed during SSR.
 *
 * Model: /model/model.json
 * Input:  [1, 40, 258]
 * Output: [1, 261] softmax probabilities
 *
 * Compatibility: Registers an 'L2' serialization alias before model load.
 *
 * Root cause: Keras 2.15.0 exports the L2 regularizer with class_name "L2",
 * but TensorFlow.js 4.x internally uses class_name "L1L2" for both L1 and L2.
 * The deserializer fails with "Unknown regularizer: L2" because no class named
 * "L2" is registered.
 *
 * Fix: Register a subclass of L1L2 with className "L2" and a fromConfig that
 * maps { l2: value } → L1L2({ l1: 0, l2: value }).  This is done once,
 * idempotently, before tf.loadLayersModel() is called.
 */

import type { ModelState } from './types'

// Lazy-imported to avoid SSR issues with TensorFlow.js
type TF = typeof import('@tensorflow/tfjs')
type LayersModel = import('@tensorflow/tfjs').LayersModel
type Tensor = import('@tensorflow/tfjs').Tensor

const MODEL_URL = '/model/model.json'
const EXPECTED_CLASSES = 261
const EXPECTED_SEQ_LEN = 40
const EXPECTED_FEAT_DIM = 258

let _tf: TF | null = null
let _model: LayersModel | null = null
let _modelState: ModelState = 'idle'
let _loadPromise: Promise<LayersModel> | null = null
/** Guard so we only register the compat class once. */
let _l2Registered = false

/** Returns the current model loading state. */
export function getModelState(): ModelState {
  return _modelState
}

/** Returns the cached model instance, or null if not yet loaded. */
export function getModel(): LayersModel | null {
  return _model
}

/**
 * Registers the "L2" regularizer class name with TensorFlow.js.
 *
 * Keras 2.x serializes the L2 regularizer as:
 *   { "class_name": "L2", "config": { "l2": <value> } }
 *
 * TF.js 4.x only registers "L1L2" (which covers both L1 and L2).
 * We construct a dynamic subclass so that:
 *   - static className === "L2"
 *   - fromConfig maps { l2: value } → L1L2({ l1: 0, l2: value })
 *
 * Using `Object.defineProperty` / prototype manipulation avoids TypeScript
 * override-modifier errors that arise when extending a dynamically-obtained
 * constructor whose base type is not statically known.
 *
 * Registration is idempotent — safe to call multiple times.
 * We only need this for inference; no training/optimizer interaction.
 */
async function registerL2Compat(tf: TF): Promise<void> {
  if (_l2Registered) return
  _l2Registered = true

  try {
    // Obtain the L1L2 constructor via the regularizers factory.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const l1l2Instance: any = tf.regularizers.l1l2({ l1: 0, l2: 1e-4 })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const L1L2Ctor: any = l1l2Instance.constructor

    // Dynamically build a subclass to avoid TypeScript override errors on an
    // untyped base. At runtime this is a valid ES6 class that extends L1L2.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const L2Regularizer = (function buildL2Class(Base: any) {
      class L2Class extends Base {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        constructor(config: any) {
          super(config)
        }

        getClassName(): string {
          return 'L2'
        }

        /**
         * Called by TF.js deserializer with the JSON config from model.json.
         * Keras L2 config: { "l2": <float> }
         * L1L2 ctor config: { l1: <float>, l2: <float> }
         */
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        static fromConfig(cls: any, config: Record<string, unknown>) {
          return new cls({ l1: 0, l2: (config['l2'] as number) ?? 0 })
        }
      }
      // className must be on the constructor itself as a static property
      Object.defineProperty(L2Class, 'className', {
        value: 'L2',
        writable: false,
        configurable: true,
      })
      return L2Class
    })(L1L2Ctor)

    tf.serialization.registerClass(
      L2Regularizer as unknown as Parameters<typeof tf.serialization.registerClass>[0]
    )
    console.log('[ISL] L2 regularizer compatibility class registered')
  } catch (err) {
    // Non-fatal: log a warning and reset guard so a future call can retry.
    console.warn('[ISL] L2 compat registration warning (non-fatal):', err)
    _l2Registered = false
  }
}

/**
 * Loads and caches the TensorFlow.js Layers model.
 * Safe to call multiple times — only loads once.
 * Validates input/output shapes after loading.
 */
export async function loadModel(): Promise<LayersModel> {
  if (_model !== null) return _model
  if (_loadPromise !== null) return _loadPromise

  _modelState = 'loading'
  console.log('[ISL] Model loading...')

  _loadPromise = (async () => {
    // Dynamic import to keep TF out of SSR
    const tf = await import('@tensorflow/tfjs')
    await tf.setBackend('cpu')
    await tf.ready()
    _tf = tf

    // ── STEP 1: Register the L2 compatibility class ──────────────────
    await registerL2Compat(tf)

    // ── STEP 2: Load the model ────────────────────────────────────────
    let model: LayersModel
    try {
      model = await tf.loadLayersModel(MODEL_URL)
    } catch (err) {
      _modelState = 'error'
      _loadPromise = null
      throw new Error(`[ISL] Failed to load model from ${MODEL_URL}: ${String(err)}`)
    }

    // ── STEP 3: Validate input shape ──────────────────────────────────
    const inputShape = model.inputs[0]?.shape
    console.log('[ISL] Input shape:', inputShape)
    if (
      inputShape == null ||
      inputShape.length !== 3 ||
      (inputShape[1] !== null && inputShape[1] !== EXPECTED_SEQ_LEN) ||
      (inputShape[2] !== null && inputShape[2] !== EXPECTED_FEAT_DIM)
    ) {
      _modelState = 'error'
      throw new Error(
        `[ISL] Unexpected model input shape: ${JSON.stringify(inputShape)}. ` +
        `Expected [null, ${EXPECTED_SEQ_LEN}, ${EXPECTED_FEAT_DIM}].`
      )
    }

    // ── STEP 4: Validate output shape ─────────────────────────────────
    const outputShape = model.outputs[0]?.shape
    if (
      outputShape == null ||
      outputShape.length !== 2 ||
      (outputShape[1] !== null && outputShape[1] !== EXPECTED_CLASSES)
    ) {
      _modelState = 'error'
      throw new Error(
        `[ISL] Unexpected model output shape: ${JSON.stringify(outputShape)}. ` +
        `Expected [null, ${EXPECTED_CLASSES}].`
      )
    }

    console.log('[ISL] Model ready. Input:', inputShape, 'Output:', outputShape)
    _model = model
    _modelState = 'ready'
    return model
  })()

  return _loadPromise
}

/**
 * Runs a single inference pass.
 *
 * @param normalizedFrames — array of WINDOW_SIZE pre-normalized Float32Array(258) frames
 * @returns Float32Array of 261 softmax probabilities
 *
 * All intermediate tensors are disposed.
 * The caller receives a plain JS Float32Array, not a tensor.
 */
export async function runInference(normalizedFrames: Float32Array[]): Promise<Float32Array> {
  if (_model === null) throw new Error('[ISL] runInference called before model is ready')
  if (_tf === null) throw new Error('[ISL] TensorFlow.js not loaded')

  const tf = _tf
  const model = _model

  const seqLen = normalizedFrames.length
  const featDim = normalizedFrames[0]?.length ?? 0

  if (seqLen !== EXPECTED_SEQ_LEN) {
    throw new Error(`[ISL] runInference: got ${seqLen} frames, expected ${EXPECTED_SEQ_LEN}`)
  }
  if (featDim !== EXPECTED_FEAT_DIM) {
    throw new Error(`[ISL] runInference: frame has ${featDim} features, expected ${EXPECTED_FEAT_DIM}`)
  }

  // Build flat data buffer
  const flat = new Float32Array(seqLen * featDim)
  for (let i = 0; i < seqLen; i++) {
    flat.set(normalizedFrames[i], i * featDim)
  }

  let inputTensor: Tensor | null = null
  let outputTensor: Tensor | null = null

  try {
    // Shape: [1, 40, 258]
    inputTensor = tf.tensor3d(flat, [1, seqLen, featDim])
    console.log('[ISL DEBUG] INPUT TENSOR SHAPE:', inputTensor.shape)

    // Run prediction — model.predict returns Tensor | Tensor[]
    console.log('[ISL DEBUG] CALLING MODEL.PREDICT')
    console.log('[ISL] predict executed')
    const raw = model.predict(inputTensor)
    outputTensor = Array.isArray(raw) ? raw[0] : raw
    console.log('[ISL DEBUG] MODEL OUTPUT SHAPE:', outputTensor.shape)

    // Extract probabilities as a plain JS Float32Array (copy, not a view)
    const probs = new Float32Array(await outputTensor.data())

    if (probs.length !== EXPECTED_CLASSES) {
      throw new Error(
        `[ISL] Model output has ${probs.length} values, expected ${EXPECTED_CLASSES}`
      )
    }

    // Validate all values are finite (guard against NaN/Infinity)
    for (let i = 0; i < probs.length; i++) {
      if (!isFinite(probs[i])) {
        throw new Error(`[ISL] Model output contains non-finite value at index ${i}: ${probs[i]}`)
      }
    }

    return probs
  } catch (err) {
    console.log('[ISL] Error in runInference: ' + (err instanceof Error ? err.message : String(err)))
    throw err
  } finally {
    // Always dispose tensors to prevent memory leaks
    inputTensor?.dispose()
    outputTensor?.dispose()
  }
}
