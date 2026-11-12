# research/ — algorithm study for the UREX tread-depth software

One simulator, one benchmark, many algorithms. Every algorithm lives on its own git branch
(`research/<topic>`) and is scored by the same harness on the same frames, so the numbers in
`results/*.json` are directly comparable.

```
research/
  sim/holder_sim.py     ray-traced simulator of the Rev A printed holder on a tyre (+ exact truth)
  bench/datasets.py     fixed test sets: sim ideal / stress / legal / train / calib, and the 86 real photos
  bench/calib.py        shim calibration: height_mm = poly(u, v)
  bench/depth.py        the default "line -> groove depth" stage
  bench/run.py          the scorer (read its docstring: API + every metric)
  bench/oracle.py       true line + default depth stage = ceiling for line extractors
  bench/real_refs.json  reference line + depth for 12 real photos ("silver": baseline-derived, eye-checked)
  algos/                one module per algorithm (baseline_steger.py is the 12 Nov pipeline)
  results/              one JSON per scored algorithm
  notes/                one write-up per branch (+ notes/figs/<branch>/)
```

## Conventions
- Images: uint8 greyscale. The laser line runs **left-right**; a deeper surface moves it **down** (larger v).
  Simulated frames are 320x240. Real photos are rotated into this convention and are 240 wide x 320 tall.
- Run everything from `research/`: `python -m bench.run algos.<module> --out results/<name>.json`.
- Data is cached in `/home/claude/urex_data` (shared, read-only for you). Real photos: `pipeline/data/photos`.

## The data
| set | frames | what | use |
|---|---|---|---|
| `ideal` | 150 | holder locked on a clean tyre in shade | test |
| `stress` | 200 | rocking +-1 deg, sunlight + gradient, wet glints, stones, zig-zag grooves, debris gaps, paint, blur, dim/saturated laser | test |
| `legal` | 150 | grooves 0.6-3.5 mm, mild conditions: the 1.6 mm pass/fail boundary | test |
| `train` | 1200 | mixed, different seed | **training only** |
| `calib` | 15 | flat plate at +6 .. -22 mm | calibration |
| real | 86 | hand-held photos 8-30 Oct; 19 usable by eye; 12 with reference line/depth | test |

Every simulated item has `on`, `off` (laser-off frame of the same scene) and `truth` (`u, v, height_mm`
of the exact line, `mask`, per-groove `depth_mm`, `u0/u1`, `floor_visible_px`, the full `scene` dict).
`sim/holder_sim.py` can render any scene you want for extra experiments (`sample_scene`, `render`).

Facts worth knowing:
- Scale: about 2.9 px/mm across the tread and ~2.1 px of line shift per mm of depth (320 px wide).
- Camera and laser both sit above the tread centre, so in an off-centre groove the wall hides part of
  the floor from both. `floor_visible_px` says how much floor is visible; grooves with < 3 px are not scored.
- Even with a perfect line the default depth stage reads ~0.07 mm too deep: the laser meets the groove
  floor a few mm further round the tyre (R = 480 mm), where the whole surface is lower.
- Real photos were hand-held with the laser not fixed to the camera. Their px values are good; mm are
  not meaningful. The 12 references come from the baseline, so the baseline scores ~perfect on them by
  construction: on real data look for *agreement*, coverage, and fewer false measurements, not "beating" it.

## Reference numbers (commit `research/base`)
| algorithm | set | line F1 | line RMSE px | depth MAE mm | P95 mm | miss | false/frame | verdict acc |
|---|---|---|---|---|---|---|---|---|
| baseline_steger | ideal | 0.931 | 0.162 | 0.147 | 0.474 | 0.090 | 0.093 | 0.980 |
| baseline_steger | stress | 0.821 | 0.235 | 1.516 | 6.678 | 0.099 | 0.280 | 0.915 |
| baseline_steger | legal | 0.977 | 0.155 | 0.053 | 0.210 | 0.054 | 0.127 | 0.993 |
| oracle_line (true line + default depth) | ideal | 0.998 | 0 | 0.071 | 0.205 | 0.011 | 0 | 0.987 |
| oracle_line | stress | 0.998 | 0 | 0.108 | 0.344 | 0.006 | 0.005 | 1.000 |
| oracle_line | legal | 1.000 | 0 | 0.018 | 0.038 | 0.044 | 0 | 1.000 |

Baseline on real: line F1 vs refs 1.000 (by construction), depth MAE 0.13 px, usable recall 0.95,
**false_measure 0.70** (no quality gate in this harness: it reports a depth on 70 % of unusable photos).

## Rules for a research branch
1. Work only in your own worktree and on your own branch. Commit locally. **Do not push, do not touch
   other worktrees, do not edit `bench/` or `sim/`.** If you think the harness is wrong, say so in your
   notes with evidence; don't patch it.
2. Add files only under `research/algos/`, `research/experiments/<topic>/`, `research/models/`,
   `research/results/`, `research/notes/`.
3. Never train or tune on `ideal`, `stress`, `legal` or the real test labels without cross-validation.
   Train on `train`; tune on a split of `train`. Report honestly if anything leaked.
4. Use libraries (numpy, scipy, scikit-learn, scikit-image, OpenCV, PyTorch are installed). Keep each
   algorithm file small and readable (a few hundred lines at most). No frameworks, no config systems.
5. The machine has 2 CPU cores shared with other agents: `export OMP_NUM_THREADS=1`, call
   `torch.set_num_threads(1)`, keep any single command under ~8 minutes, use `--limit` while developing.
   Saved models must be small (< 5 MB) and committed under `research/models/`.
6. Deliver:
   - `results/<name>.json` from `bench.run --out` for each final variant (a handful, not dozens);
   - `notes/<topic>.md`: what you tried, one results table (sim ideal / stress / legal + real), what
     worked, what failed and why, limitations, and the 1-2 code excerpts (10-25 lines) that carry the idea;
   - 2-4 figures as PNG (<= 150 dpi, light background) in `notes/figs/<topic>/`, referenced from the notes;
   - a final commit with a clear message.
7. Report numbers exactly as the harness prints them. Negative results are results: write them down.
