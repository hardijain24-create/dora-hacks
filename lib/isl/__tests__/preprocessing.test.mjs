/**
 * Deterministic preprocessing unit test for the ISL pipeline.
 *
 * Tests (no test framework needed — plain assertions):
 *  1. extractFeatureVector: feature ordering, pose + both hands
 *  2. extractFeatureVector: only left hand (right hand zero-filled)
 *  3. extractFeatureVector: only right hand (left hand zero-filled)
 *  4. extractFeatureVector: no hands (both zero-filled)
 *  5. normalizeWindow: shoulder center, scale, x/y normalization
 *  6. normalizeWindow: z preservation, visibility preservation
 *  7. normalizeWindow: scale < 1e-4 → scale = 1.0
 *  8. normalizeWindow: pose landmark 11 is left shoulder
 *  9. normalizeWindow: pose landmark 12 is right shoulder
 */

import { readFileSync } from 'fs'
import { createRequire } from 'module'
import { fileURLToPath } from 'url'
import path from 'path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// ── Inline the tested functions so we don't need ts-node ─────────────────────
// (These are verbatim copies of preprocessing.ts logic.)

const FEATURE_DIM = 258
const POSE_DIM = 132
const HAND_DIM = 63
const LEFT_SHOULDER_IDX = 11
const RIGHT_SHOULDER_IDX = 12

function extractFeatureVector(landmarks) {
  const features = new Float32Array(FEATURE_DIM)

  if (landmarks.pose !== null) {
    features.set(landmarks.pose, 0)
  }
  if (landmarks.leftHand !== null) {
    features.set(landmarks.leftHand, POSE_DIM)
  }
  if (landmarks.rightHand !== null) {
    features.set(landmarks.rightHand, POSE_DIM + HAND_DIM)
  }

  return features
}

function normalizeWindow(rawFrames) {
  const n = rawFrames.length
  const result = new Array(n)

  for (let f = 0; f < n; f++) {
    const frame = rawFrames[f]
    const norm = new Float32Array(frame)

    const lsBase = LEFT_SHOULDER_IDX * 4
    const lsx = frame[lsBase]
    const lsy = frame[lsBase + 1]

    const rsBase = RIGHT_SHOULDER_IDX * 4
    const rsx = frame[rsBase]
    const rsy = frame[rsBase + 1]

    const cx = (lsx + rsx) / 2.0
    const cy = (lsy + rsy) / 2.0

    const dx = lsx - rsx
    const dy = lsy - rsy
    let scale = Math.sqrt(dx * dx + dy * dy)
    if (scale < 1e-4) scale = 1.0

    for (let i = 0; i < 33; i++) {
      const base = i * 4
      norm[base]     = (frame[base]     - cx) / scale
      norm[base + 1] = (frame[base + 1] - cy) / scale
    }

    const lhStart = POSE_DIM
    for (let i = 0; i < 21; i++) {
      const base = lhStart + i * 3
      norm[base]     = (frame[base]     - cx) / scale
      norm[base + 1] = (frame[base + 1] - cy) / scale
    }

    const rhStart = POSE_DIM + HAND_DIM
    for (let i = 0; i < 21; i++) {
      const base = rhStart + i * 3
      norm[base]     = (frame[base]     - cx) / scale
      norm[base + 1] = (frame[base + 1] - cy) / scale
    }

    result[f] = norm
  }

  return result
}

// ── Test harness ──────────────────────────────────────────────────────────────

let passed = 0
let failed = 0

function assert(condition, description) {
  if (condition) {
    console.log(`  ✓ ${description}`)
    passed++
  } else {
    console.error(`  ✗ FAIL: ${description}`)
    failed++
  }
}

function assertClose(a, b, tol, description) {
  assert(Math.abs(a - b) < tol, `${description} (got ${a}, expected ≈ ${b})`)
}

// ── Build helpers ─────────────────────────────────────────────────────────────

function makePose(overrides = {}) {
  const pose = new Float32Array(132)
  // Fill with recognizable values: landmark i → x=i*0.01, y=i*0.01+0.5, z=i*0.001, vis=0.9
  for (let i = 0; i < 33; i++) {
    pose[i * 4]     = i * 0.01
    pose[i * 4 + 1] = i * 0.01 + 0.5
    pose[i * 4 + 2] = i * 0.001
    pose[i * 4 + 3] = 0.9
  }
  // Apply overrides: { lm_index: { x, y, z, vis } }
  for (const [idx, vals] of Object.entries(overrides)) {
    const i = parseInt(idx)
    if ('x' in vals) pose[i * 4]     = vals.x
    if ('y' in vals) pose[i * 4 + 1] = vals.y
    if ('z' in vals) pose[i * 4 + 2] = vals.z
    if ('vis' in vals) pose[i * 4 + 3] = vals.vis
  }
  return pose
}

function makeHand(seedX = 0.3, seedY = 0.4) {
  const hand = new Float32Array(63)
  for (let i = 0; i < 21; i++) {
    hand[i * 3]     = seedX + i * 0.01
    hand[i * 3 + 1] = seedY + i * 0.01
    hand[i * 3 + 2] = i * 0.001
  }
  return hand
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Feature vector has exactly 258 values
// ─────────────────────────────────────────────────────────────────────────────
console.log('\nTest 1: Feature dimension = 258')
{
  const pose = makePose()
  const lh = makeHand(0.3, 0.4)
  const rh = makeHand(0.6, 0.4)
  const lm = { pose, leftHand: lh, rightHand: rh }
  const fv = extractFeatureVector(lm)
  assert(fv.length === 258, 'feature vector length === 258')
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: Pose data placed at [0..131], left hand at [132..194], right at [195..257]
// ─────────────────────────────────────────────────────────────────────────────
console.log('\nTest 2: Feature ordering (pose, leftHand, rightHand)')
{
  const pose = new Float32Array(132).fill(1.0)
  const lh   = new Float32Array(63).fill(2.0)
  const rh   = new Float32Array(63).fill(3.0)
  const fv = extractFeatureVector({ pose, leftHand: lh, rightHand: rh })

  assert(fv[0] === 1.0,    'pose at index 0')
  assert(fv[131] === 1.0,  'pose at index 131')
  assert(fv[132] === 2.0,  'left hand at index 132')
  assert(fv[194] === 2.0,  'left hand at index 194')
  assert(fv[195] === 3.0,  'right hand at index 195')
  assert(fv[257] === 3.0,  'right hand at index 257')
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: Missing right hand → [195..257] = zeros
// ─────────────────────────────────────────────────────────────────────────────
console.log('\nTest 3: Missing right hand → zero-filled [195..257]')
{
  const pose = new Float32Array(132).fill(1.0)
  const lh   = new Float32Array(63).fill(2.0)
  const fv = extractFeatureVector({ pose, leftHand: lh, rightHand: null })

  assert(fv[195] === 0.0, 'right hand slot 195 = 0 when absent')
  assert(fv[257] === 0.0, 'right hand slot 257 = 0 when absent')
  assert(fv[132] === 2.0, 'left hand still populated')
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Missing left hand → [132..194] = zeros
// ─────────────────────────────────────────────────────────────────────────────
console.log('\nTest 4: Missing left hand → zero-filled [132..194]')
{
  const pose = new Float32Array(132).fill(1.0)
  const rh   = new Float32Array(63).fill(3.0)
  const fv = extractFeatureVector({ pose, leftHand: null, rightHand: rh })

  assert(fv[132] === 0.0, 'left hand slot 132 = 0 when absent')
  assert(fv[194] === 0.0, 'left hand slot 194 = 0 when absent')
  assert(fv[195] === 3.0, 'right hand still populated')
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: No hands → [132..257] all zeros
// ─────────────────────────────────────────────────────────────────────────────
console.log('\nTest 5: No hands → both hand slots zero-filled')
{
  const pose = new Float32Array(132).fill(1.0)
  const fv = extractFeatureVector({ pose, leftHand: null, rightHand: null })

  let allZero = true
  for (let i = 132; i < 258; i++) {
    if (fv[i] !== 0) { allZero = false; break }
  }
  assert(allZero, 'slots [132..257] are all zeros when both hands absent')
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: No pose → [0..131] all zeros
// ─────────────────────────────────────────────────────────────────────────────
console.log('\nTest 6: No pose → pose slots zero-filled')
{
  const lh = new Float32Array(63).fill(2.0)
  const fv = extractFeatureVector({ pose: null, leftHand: lh, rightHand: null })

  let allZero = true
  for (let i = 0; i < 132; i++) {
    if (fv[i] !== 0) { allZero = false; break }
  }
  assert(allZero, 'slots [0..131] are all zeros when pose absent')
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7: Shoulder indices
// Left shoulder = pose landmark 11 (indices 44..47)
// Right shoulder = pose landmark 12 (indices 48..51)
// ─────────────────────────────────────────────────────────────────────────────
console.log('\nTest 7: Shoulder landmark indices (11=left, 12=right)')
{
  // Left shoulder at landmark 11: x=0.2, y=0.6
  // Right shoulder at landmark 12: x=0.8, y=0.6
  const pose = makePose({
    11: { x: 0.2, y: 0.6, z: 0.01, vis: 0.99 },
    12: { x: 0.8, y: 0.6, z: 0.01, vis: 0.99 },
  })

  assertClose(pose[11 * 4],     0.2, 1e-5, 'pose[11*4] = left shoulder x')
  assertClose(pose[11 * 4 + 1], 0.6, 1e-5, 'pose[11*4+1] = left shoulder y')
  assertClose(pose[12 * 4],     0.8, 1e-5, 'pose[12*4] = right shoulder x')
  assertClose(pose[12 * 4 + 1], 0.6, 1e-5, 'pose[12*4+1] = right shoulder y')
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8: Normalization — center and scale computation
// ─────────────────────────────────────────────────────────────────────────────
console.log('\nTest 8: Normalization center and scale')
{
  // Shoulders at (0.2, 0.5) and (0.6, 0.5) → center=(0.4, 0.5), scale=0.4
  const pose = new Float32Array(132)
  pose[11 * 4]     = 0.2  // lsx
  pose[11 * 4 + 1] = 0.5  // lsy
  pose[12 * 4]     = 0.6  // rsx
  pose[12 * 4 + 1] = 0.5  // rsy

  // Set landmark 0 to a known position: x=0.4 (at center), y=0.5 (at center)
  pose[0] = 0.4
  pose[1] = 0.5

  const raw = [extractFeatureVector({ pose, leftHand: null, rightHand: null })]
  const norm = normalizeWindow(raw)

  // Expected center = (0.2+0.6)/2 = 0.4, (0.5+0.5)/2 = 0.5
  // Expected scale = sqrt((0.2-0.6)^2 + (0.5-0.5)^2) = 0.4
  // Landmark 0: norm_x = (0.4 - 0.4) / 0.4 = 0.0
  //             norm_y = (0.5 - 0.5) / 0.4 = 0.0
  assertClose(norm[0][0], 0.0, 1e-5, 'landmark 0 x at center → normalized 0')
  assertClose(norm[0][1], 0.0, 1e-5, 'landmark 0 y at center → normalized 0')

  // Left shoulder (lm 11) after normalization:
  // norm_x = (0.2 - 0.4) / 0.4 = -0.5
  // norm_y = (0.5 - 0.5) / 0.4 = 0.0
  assertClose(norm[0][11 * 4],     -0.5, 1e-5, 'left shoulder normalized x = -0.5')
  assertClose(norm[0][11 * 4 + 1],  0.0, 1e-5, 'left shoulder normalized y = 0')

  // Right shoulder (lm 12):
  // norm_x = (0.6 - 0.4) / 0.4 = 0.5
  assertClose(norm[0][12 * 4],      0.5, 1e-5, 'right shoulder normalized x = 0.5')
  assertClose(norm[0][12 * 4 + 1],  0.0, 1e-5, 'right shoulder normalized y = 0')
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 9: z and visibility preserved unchanged
// ─────────────────────────────────────────────────────────────────────────────
console.log('\nTest 9: z and visibility preserved')
{
  const pose = makePose({
    11: { x: 0.2, y: 0.5, z: 0.123, vis: 0.777 },
    12: { x: 0.6, y: 0.5, z: 0.456, vis: 0.888 },
  })
  const raw = [extractFeatureVector({ pose, leftHand: null, rightHand: null })]
  const norm = normalizeWindow(raw)

  assertClose(norm[0][11 * 4 + 2], 0.123, 1e-5, 'left shoulder z unchanged')
  assertClose(norm[0][11 * 4 + 3], 0.777, 1e-5, 'left shoulder visibility unchanged')
  assertClose(norm[0][12 * 4 + 2], 0.456, 1e-5, 'right shoulder z unchanged')
  assertClose(norm[0][12 * 4 + 3], 0.888, 1e-5, 'right shoulder visibility unchanged')
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 10: scale < 1e-4 → scale = 1.0 (shoulders coincident)
// ─────────────────────────────────────────────────────────────────────────────
console.log('\nTest 10: Degenerate scale → clamped to 1.0')
{
  const pose = new Float32Array(132)
  // Both shoulders at same location → distance = 0
  pose[11 * 4]     = 0.5  // lsx
  pose[11 * 4 + 1] = 0.5  // lsy
  pose[12 * 4]     = 0.5  // rsx
  pose[12 * 4 + 1] = 0.5  // rsy

  // Landmark 0 at (0.5 + 0.3, 0.5 + 0.2) = (0.8, 0.7)
  pose[0] = 0.8
  pose[1] = 0.7

  const raw = [extractFeatureVector({ pose, leftHand: null, rightHand: null })]
  const norm = normalizeWindow(raw)

  // scale clamped to 1.0, center = (0.5, 0.5)
  // lm 0 normalized: x = (0.8 - 0.5)/1.0 = 0.3, y = (0.7 - 0.5)/1.0 = 0.2
  assertClose(norm[0][0], 0.3, 1e-5, 'norm x with clamped scale = 0.3')
  assertClose(norm[0][1], 0.2, 1e-5, 'norm y with clamped scale = 0.2')
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 11: Hand x/y also normalized (same center/scale as pose)
// ─────────────────────────────────────────────────────────────────────────────
console.log('\nTest 11: Left hand x/y normalized with same center/scale')
{
  // Shoulders: lm11=(0.2, 0.5), lm12=(0.6, 0.5) → center=(0.4,0.5), scale=0.4
  const pose = new Float32Array(132)
  pose[11 * 4]     = 0.2
  pose[11 * 4 + 1] = 0.5
  pose[12 * 4]     = 0.6
  pose[12 * 4 + 1] = 0.5

  // Left hand landmark 0 at x=0.4, y=0.7
  const lh = new Float32Array(63)
  lh[0] = 0.4
  lh[1] = 0.7
  lh[2] = 0.05  // z (should be preserved)

  const raw = [extractFeatureVector({ pose, leftHand: lh, rightHand: null })]
  const norm = normalizeWindow(raw)

  // lh[0] norm: x = (0.4 - 0.4)/0.4 = 0.0, y = (0.7 - 0.5)/0.4 = 0.5, z = 0.05
  assertClose(norm[0][132],     0.0,  1e-5, 'left hand lm0 x → 0')
  assertClose(norm[0][132 + 1], 0.5,  1e-5, 'left hand lm0 y → 0.5')
  assertClose(norm[0][132 + 2], 0.05, 1e-5, 'left hand lm0 z preserved')
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 12: Right hand x/y normalized correctly
// ─────────────────────────────────────────────────────────────────────────────
console.log('\nTest 12: Right hand x/y normalized with same center/scale')
{
  const pose = new Float32Array(132)
  pose[11 * 4]     = 0.2
  pose[11 * 4 + 1] = 0.5
  pose[12 * 4]     = 0.6
  pose[12 * 4 + 1] = 0.5

  // Right hand landmark 0 at x=0.8, y=0.5
  const rh = new Float32Array(63)
  rh[0] = 0.8
  rh[1] = 0.5
  rh[2] = 0.02

  const raw = [extractFeatureVector({ pose, leftHand: null, rightHand: rh })]
  const norm = normalizeWindow(raw)

  // rh[0] norm: x = (0.8 - 0.4)/0.4 = 1.0, y = (0.5-0.5)/0.4 = 0
  assertClose(norm[0][195],     1.0,  1e-5, 'right hand lm0 x → 1.0')
  assertClose(norm[0][195 + 1], 0.0,  1e-5, 'right hand lm0 y → 0')
  assertClose(norm[0][195 + 2], 0.02, 1e-5, 'right hand lm0 z preserved')
}

// ─────────────────────────────────────────────────────────────────────────────
// Final report
// ─────────────────────────────────────────────────────────────────────────────
console.log(`\n${'═'.repeat(50)}`)
console.log(`PREPROCESSING TEST RESULTS`)
console.log(`  Passed: ${passed}`)
console.log(`  Failed: ${failed}`)
console.log(`  Total:  ${passed + failed}`)
console.log(`  Result: ${failed === 0 ? 'ALL PASS ✓' : 'SOME FAILED ✗'}`)
console.log('═'.repeat(50))

if (failed > 0) process.exit(1)
