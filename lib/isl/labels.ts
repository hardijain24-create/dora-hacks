/**
 * ISL Recognition Pipeline — Label Service
 *
 * Loads /model/labels.json exactly once.
 * Preserves the exact array order from the file (training sort order).
 * Provides a display-clean version of each label.
 *
 * Training notebook (Cell 8):
 *   classes = sorted(set(y_labels))
 *   label_to_idx = {label: i for i, label in enumerate(classes)}
 * Therefore labels.json order == output index order.
 */

const EXPECTED_LABEL_COUNT = 261

let _labels: string[] | null = null
let _loadPromise: Promise<string[]> | null = null

/**
 * Loads labels exactly once and caches them.
 * Throws if the file is missing or the label count is wrong.
 */
export async function loadLabels(): Promise<string[]> {
  if (_labels !== null) return _labels
  if (_loadPromise !== null) return _loadPromise

  _loadPromise = (async () => {
    const res = await fetch('/model/labels.json')
    if (!res.ok) {
      throw new Error(`[ISL] Failed to fetch /model/labels.json: ${res.status} ${res.statusText}`)
    }
    const data: unknown = await res.json()
    if (!Array.isArray(data) || data.some(x => typeof x !== 'string')) {
      throw new Error('[ISL] labels.json must be a string array')
    }
    const labels = data as string[]
    if (labels.length !== EXPECTED_LABEL_COUNT) {
      throw new Error(
        `[ISL] labels.json has ${labels.length} entries but expected ${EXPECTED_LABEL_COUNT}. ` +
        'The label file does not match the model. Stopping to avoid incorrect predictions.'
      )
    }
    console.log(`[ISL] Labels loaded: ${labels.length}`)
    _labels = labels
    return labels
  })()

  return _loadPromise
}

/**
 * Returns the cached label array. Call loadLabels() first.
 * Returns null if labels are not yet loaded.
 */
export function getLabels(): string[] | null {
  return _labels
}

/**
 * Strips the numeric prefix from a raw label for display only.
 *
 * Training format examples:
 *   "48. Hello"  → "Hello"
 *   "41. Shirt"  → "Shirt"
 *   "88. cold"   → "cold"
 *   "Extra"      → "Extra"   (no prefix → returned as-is)
 *
 * The underlying labels array is never modified.
 */
export function cleanLabel(rawLabel: string): string {
  return rawLabel.replace(/^\s*\d+\.\s*/, '').trim()
}

/**
 * Returns the display label for a given prediction index.
 * Returns null if index is out of range or labels not loaded.
 */
export function getLabelForIndex(index: number): string | null {
  if (_labels === null) return null
  if (index < 0 || index >= _labels.length) {
    console.error(`[ISL] getLabelForIndex: index ${index} is out of range [0, ${_labels.length})`)
    return null
  }
  return cleanLabel(_labels[index])
}
