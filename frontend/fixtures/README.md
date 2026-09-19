Synthetic stand-ins for `results/*.json`, in the exact shapes `evaluate.py` and `cli.py`
write. They exist so the page can be built and reviewed before the pipeline has run.

**No number in here is a measurement.** Regenerate with `node make-fixtures.mjs`.
`npm run demo` serves this directory and shows a banner saying so; `npm run dev` reads
the real `results/`.

Demo data must never be written into `results/`: `cli.py _require_gates_open()` freezes
every gate once any `results/*.json` other than `gates.json` and `deviations.json` exists.
