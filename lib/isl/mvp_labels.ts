/**
 * MVP labels are stored separately from INCLUDE labels.
 * /model/labels.json         = INCLUDE model labels (261 classes, untouched)
 * /models/isl-mvp/labels.json = MVP model labels (created after training)
 */
const MVP_LABELS_URL = '/models/isl-mvp/labels.json'

let mvpLabels: string[] | null = null

export async function loadMvpLabels(): Promise<boolean> {
  if (mvpLabels !== null) return true

  try {
    const res = await fetch(MVP_LABELS_URL)
    if (!res.ok) {
      console.warn(`[ISL MVP] Labels not found at ${MVP_LABELS_URL} — MVP model not yet trained.`)
      return false
    }
    const data = await res.json()
    if (!Array.isArray(data) || data.length === 0) {
      console.warn('[ISL MVP] Label array is invalid or empty.')
      return false
    }
    mvpLabels = data
    console.log(`[ISL MVP] labels loaded: ${mvpLabels.length}`)
    return true
  } catch (err) {
    console.warn('[ISL MVP] Label loading error:', err)
    return false
  }
}

export function getMvpLabelForIndex(idx: number): string {
  if (!mvpLabels) return 'UNKNOWN'
  if (idx < 0 || idx >= mvpLabels.length) return 'UNKNOWN'
  return mvpLabels[idx]
}
