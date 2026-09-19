Synthetic stand-ins for `results/*.json`, in the exact shapes `evaluate.py` and `cli.py` write,
so the page can be built and reviewed before the pipeline has run.

**No number they contain is a measurement.** They are generated, not committed: `npm run demo`
writes them and serves this directory behind a banner saying so, and every generated file carries
a `_synthetic` marker. `npm run dev` reads the real `results/`.

Demo data must never be written into `results/`: `cli.py _require_gates_open()` freezes every gate
once any `results/*.json` other than `gates.json` and `deviations.json` exists, which would block
G2c and, with it, the whole pipeline.
