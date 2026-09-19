import { useEffect, useState } from 'react'

// Every file the page can show. A missing one is a stage that has not run yet, not an error.
export const FILES = [
  'summary',
  'gates',
  'deviations',
  'holdout',
  'chrono',
  'chrono_null',
  'tfidf',
  'tfidf_null',
  'vader',
  'old_bug_vader',
]

// Python's json.dumps writes a bare NaN for a non-finite float, which JSON.parse rejects.
// These files hold numbers, model names and timestamps - no free text - so replacing the
// bare tokens cannot corrupt a string. Upgrade path: make cli.py _write_json emit null
// (and report.py _fmt handle it), then drop this.
const parse = (text) => JSON.parse(text.replace(/\b(NaN|-?Infinity)\b/g, 'null'))

async function loadOne(name) {
  const response = await fetch(`./${name}.json`, { cache: 'no-store' })
  if (!response.ok) return null
  try {
    return parse(await response.text())
  } catch {
    return null // a half-written file during a run
  }
}

export function useResults() {
  const [state, setState] = useState({ loading: true, files: {} })

  useEffect(() => {
    let live = true
    Promise.all(FILES.map(loadOne)).then((loaded) => {
      if (!live) return
      setState({
        loading: false,
        files: Object.fromEntries(FILES.map((name, i) => [name, loaded[i]])),
      })
    })
    return () => {
      live = false
    }
  }, [])

  return state
}

export const isNumber = (v) => typeof v === 'number' && Number.isFinite(v)

export const fmt = (v, digits = 4) => (isNumber(v) ? v.toFixed(digits) : 'n/a')

export const fmtInt = (v) => (isNumber(v) ? v.toLocaleString('en-US') : 'n/a')
