let mvpLabels: string[] | null = null

export async function loadMvpLabels(): Promise<void> {
  if (mvpLabels !== null) return

  try {
    const res = await fetch('/model/labels.json')
    if (!res.ok) {
      throw new Error(`[ISL MVP] Failed to fetch mvp-labels.json: ${res.statusText}`)
    }
    const data = await res.json()
    if (!Array.isArray(data) || data.length === 0) {
      throw new Error('[ISL MVP] Label array is invalid or empty.')
    }
    mvpLabels = data
    console.log(`[ISL MVP] labels loaded: ${mvpLabels.length}`)
  } catch (err) {
    console.error('[ISL MVP] Label loading error:', err)
    throw err
  }
}

export function getMvpLabelForIndex(idx: number): string {
  if (!mvpLabels) return 'UNKNOWN'
  if (idx < 0 || idx >= mvpLabels.length) return 'UNKNOWN'
  return mvpLabels[idx]
}
