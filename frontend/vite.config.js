import { resolve } from 'node:path'

import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The page reads results/*.json directly: Vite serves publicDir in dev and copies it into
// dist on build, so there is no sync step and no backend.
//
// `--mode demo` serves frontend/fixtures instead. Demo data must never be written into
// results/: cli.py _require_gates_open() freezes every gate once any results/*.json other
// than gates.json and deviations.json exists, which would block G2c.
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  base: './', // GitHub Pages serves from a subpath
  publicDir:
    mode === 'demo'
      ? resolve(import.meta.dirname, 'fixtures')
      : resolve(import.meta.dirname, '../results'),
}))
