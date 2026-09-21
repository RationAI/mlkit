# PROV work — session handoff (2026-09-21)

Reconstructed from the transcript of session `be5dcd41-c69a-4dbf-a59b-28d6090c1108`
("Prov branch"), which died at 08:35 today on `ContextWindowExceededError`
(347k tokens vs a 262k window) while reading a rendered PNG. `/compact` failed
for the same reason — summarising needs the whole context too.

**Read [PROV_TODO.md](PROV_TODO.md) first.** It is the authoritative plan and
still accurate: meeting decisions D1–D7, findings A–F, tiers T1–T14. This file
covers only what happened *after* it was written, plus what a new session needs
to orient.

## Where things stand

T1–T6 are **done on the assembler side** (`provenance.py`). Tier 2's mlkit half
is deliberately untouched — see finding B there: `provenance.py` rebuilds the
graph from MLflow params/tags and never reads per-run `prov.json`, so edits to
`rationai/mlkit/provenance/` are invisible in the graph.

Verified in this sandbox just now:

- `python provenance.py --self-check` → passes
- `ruff check provenance.py render_fallback_png.py` → clean
- `mypy render_fallback_png.py` → clean
- `mypy --strict provenance.py` → 31 errors, all pre-existing debt (bare `dict`
  generics, untyped `Probe` methods). None in T1–T6 code — checked by AST pass.

## The last thing in flight: PNG rendering without graphviz

That session's final task was producing a real image. It could not install
graphviz `dot` (see below), so it built a pure-Python fallback instead:

- [render_fallback_png.py](render_fallback_png.py) — parses the `.dot` that
  `provenance.py` emits (via `pydot`), builds a `networkx.DiGraph`, and renders
  a layered layout with matplotlib. 201 lines, lint- and mypy-clean.
- It works: `output/prov-train_vgg16.png` (199 KB, 2295×1485) was written at
  08:35 by `render_fallback_png.py output/prov-train_vgg16.dot`.
- **The one thing left undone: nobody has looked at that image and said whether
  it is readable.** The session died at the moment of viewing it. An earlier
  crude pass produced a "meaningless layout"; this layered version is the fix
  for that, but it is unconfirmed visually.

Undeclared dependency, worth a decision: `matplotlib` + `pillow` were installed
into `.venv` with `uv pip install` for this renderer and are **not in
`pyproject.toml`**. Either add them to the `prov` dependency group or accept the
renderer as an ad-hoc tool.

## Untracked files, and which ones matter

`provenance.py` is still untracked — its home (`tools/`?) is an open question
from T1. Everything else untracked is scaffolding from a local end-to-end run
against `output/local_mlruns` (a file-store MLflow experiment, used because the
real tracking server is unreachable):

| path | what it is | keep? |
| --- | --- | --- |
| `provenance.py` | the assembler script, all of T1–T6 | yes, needs a home |
| `render_fallback_png.py` | PNG fallback renderer | yes |
| `PROV_TODO.md` | the plan | yes |
| `output/` | generated `.dot`/`.provn`/`.png`/`.json` + `local_mlruns/` | no — generated |
| `model/` | 2-file fake MLmodel fixture (`MLmodel` + `weights.pt`, 64 B total) | fixture, or delete |
| `tiles/train.csv` | 3-line fake tile index | fixture, or delete |
| `provenance_b9db50….prov.json` | per-run PROV doc from the local run | no — generated |

`output/`, `model/`, `tiles/` and `*.prov.json` are prime `.gitignore`
candidates before anything gets staged.

## Two cheap questions to ask humans

Unchanged from PROV_TODO.md's closing note, and still the highest-value next
moves because both block code rather than merely improving it:

1. **Which namespace version defines `BundleMetadata`,
   `referencedMetaBundleId`, `referencedMetaspectVersion`?** None appear on the
   published CPM namespace, yet we emit them under it (finding D). Decides
   whether T3's placeholders are values to fill or names to replace.
2. **Should the per-run main activities in mlkit be dropped?** Per D1 one main
   activity belongs over the chain, so the per-run ones are redundant — and the
   mlkit halves of T5/T6/T2a are mechanical once that is decided.

Plus the coordination asks: T13 (is "create dataset" ours or MOU's) before T8
starts, and T14 (what id will MOU mint, and what will their `backwardConnector`
point at — it must target our `cpm:forwardConnector`'s exact id).

## Next code work

**T7** (dataset entity + explicit `used`), gated on decision **T13a** — one
dataset entity or two (index + on-disk images)? Then **T10** (sha256 in
`register_dataset`; under D4 this is the mechanism binding the graph to real
data, and today `verify_manifest` compares file *sizes* only, so a same-length
edit still reports `VERIFIED`). **T10b** is the final acceptance check: can the
graph answer "on which files was this model trained?" Today, no.

## Environment constraints (re-verified)

No graphviz `dot`, not installable here (no sudo; a `brew install` attempt was
rejected — don't retry without asking). Tracking server
`mlflow-jiribuchta.dyn.cloud.trusted.e-infra.cz` unreachable; PyPI and general
egress work. So `--self-check` plus a **real PNG via `render_fallback_png.py`**
are the only verification possible here — a real *chain* still needs the user's
machine. Never present a real-chain result as verified from this sandbox.
