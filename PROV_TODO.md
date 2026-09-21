# PROV graph — TODO

Ranked easiest → hardest. `[REPO]` = code change here. `[BLOCKED]` = needs an
input we lack. `[DECISION]` = needs a human call before code. `[NOT CODE]` =
org/process, tracked so nothing is dropped.

Updated with the meeting decisions. Meeting-derived changes are marked **(M)**.

## Meeting decisions that reshape the plan

**D1. The main activity is NOT the training activity.** It represents what the
team is responsible for and sits **over the whole chain**, so the chain's end
(the trained model) can be referenced from it.
→ Current code is wrong on this in two ways, and both are cheap to fix:
- `provenance.py:359` sets `dct:hasPart: [gen:run_<root_id>]` — the main activity
  parts only the **root run**, so upstream preprocessing runs are outside it.
  Should reference the whole chain.
- `run.py:407` builds a main activity per training run (`blank:TrainingRun_*`),
  i.e. one per run rather than one over the chain. Per-run builders cannot
  express a chain-level main activity — it belongs to the assembler only.

**D2. Identifiers must match exactly or they are different objects.** This is
live now, and it answers the meeting's open question about `gen:run`:
| where | main activity id |
| --- | --- |
| `provenance.py:355` | `blank:Run_<root_id[:10]>` |
| `log_provenance.py:295` | `blank:Run_<run_id[:8]>` |
| `run.py:407` | `blank:TrainingRun_<run_id[:8]>` |

Three spellings of "the same" node, and two different truncation lengths (10 vs
8). None of them can join. New item **T2a**.

**D3. Two dimensions must not be mixed in one graph** — model/dataset history vs.
**how the provenance graph itself was generated** (the latter excluded). The
assembler is external, so it is fine; but `register_dataset`/`register_new_user`
each emit their own `prov.json` (`dataset.py:503`, `user.py:216`), so
registration *is* currently graph content. Open for **T10**.

**D4. The graph must bind to the real data, not just its metadata description.**
This raises the stakes on hashing (T10) and on T7's input entity: today the
assembler's output entities are `sosa:Sample` whose `schema:url` is
`run.info.artifact_uri` — a pointer at MLflow storage, not at the VSI files.

**D5. Training takes TWO inputs: the generated index + the real images on disk.**
The assembler only discovers `mlflow-artifacts:/` URIs (`URI_RE`,
`provenance.py:46`). **Real images are a local path, so they are structurally
invisible to the current chain builder.** This is not a modelling nicety — the
"dataset" input the meeting asks for is exactly the part the regex cannot see.
Feeds **T7** and **T8**.

**D6. MOU linkage is deferred**; the chain will legitimately appear to start at
our end. T14 stays as a question to ask, not work to do now.

**D7. Don't count on MOU source data** — Matúš cannot access it. **T8 must be
built and verified against synthetic/dummy data**, not the real VSI cohort.

## Findings from implementing T5/T6

**C. The CPM vocabulary is public — T9 was never fully blocked.** The namespace
and the backbone template are both published and reachable from this sandbox:
<https://www.commonprovenancemodel.org/cpm-namespace/> and
<https://www.commonprovenancemodel.org/cpm-backbone-template-v1/>. The namespace
defines, with these exact spellings: `currentConnector`, `forwardConnector`,
`backwardConnector`, `jumpForwardConnector`, `jumpBackwardConnector`,
`mainActivity`, `receiptActivity`, `senderAgent`, `receiverAgent`,
`senderBundleId`, `receiverBundleId`, `currentBundle`, `metabundle`. So the
`CPM_CONNECTORS = {"forward": "cpm:TODO-forwardConnector"}` scaffolding in
`provenance.py` was guessing at something we could have looked up. T5/T6 are now
built on the real terms.

**D. Three terms this repo emits are NOT in that namespace.** `BundleMetadata`,
`referencedMetaBundleId`, `referencedMetaspectVersion` — zero occurrences on the
namespace page, and the closest defined term is `metabundle`. The namespace URI
in `common.py:22` and `provenance.py:57` is the CPM's own, so these three are
being published under a namespace that does not define them. Consequences:
- T3's "mandatory main-activity attributes" are not mandatory per the published
  CPM; whoever specified them either has a newer namespace version or a local
  extension. **Ask which** — this is a cheaper question than T9's full document
  request, and it is the one thing that can make T3's placeholders wrong rather
  than merely placeholder-y.
- The two placeholders stay as-is (D4 decision), but they are now known to be
  non-standard names, not just unfilled values.
- The `cpm:` prefixes this task *did* verify (`mainActivity`, `forwardConnector`,
  `currentConnector`) need no such caveat.

**E. The backbone is only half-emittable from inside mlkit, by construction.**
The template's traversal chain is
`backwardConnector → receiptActivity → currentConnector → mainActivity →
forwardConnector`. `receiptActivity` and `backwardConnector` describe the *other*
party's step (the receiver of our output). Emitting them from mlkit would mean
asserting something about a system we do not observe, so the assembler emits the
sending half only: `mainActivity` + `currentConnector`(s) + `forwardConnector`.
Stated here because a reviewer holding the reference figure will otherwise ask
where the other two nodes are.

**F. `wasDerivedFrom` field names in the JSON path do not match PROV-JSON.**
Pre-existing: the JSON export writes `prov:entity` / `prov:derivation`, while the
[PROV-JSON grammar](https://www.w3.org/Submission/2013/SUBM-prov-json-20130424/)
specifies `prov:usedEntity` / `prov:generatedEntity`. Not fixed here — fixing it
means changing every existing `wasDerivedFrom` record and could break the Java
`prov_mlflow` consumer this format exists to match. T5's new derivation records
therefore follow the *existing* spelling on purpose; a document half in each
convention is worse than one consistently in the other. Flag for whoever owns the
Java-side contract. (Checked in the same pass: `used`, `wasGeneratedBy`,
`wasAttributedTo`, `wasAssociatedWith` field names **are** correct, and
`qualifiedAssociation` is not a PROV-JSON section at all — another Java-ism.)

---

## Two code findings that still stand

**A. The two export paths have already diverged.** `build_provenance()` (→ JSON)
emits `cpm:mainActivity`, `cpm:BundleMetadata`, and the param/environment
`prov:Annotation`s; `build_prov_document()` (→ **PNG** and `.provn`) emits
**none** — verified, no `blank:`/`meta:`/`cpm:`/annotation code in its body
(`provenance.py:390-439`). **This fully explains the JSON-vs-diagram discrepancy**
— not a renderer bug, not a stale picture. The PNG cannot contain the node the
document never had.

**B. `provenance.py` never reads the per-run `provenance/prov.json` files** — it
rebuilds the chain from MLflow params/tags. So fixes to the builders in
`rationai/mlkit/provenance/` are invisible in the graph. Every structural change
is therefore a change to `provenance.py` (both functions, per T2 note below).

Duplication stays by decision, so each structural change is made **twice** and
`--self-check` (`provenance.py:568-610`) is the guard.

**Transcription note:** "Hespart" in the summary is almost certainly
**`dct:hasPart`** — the property the main activity already uses
(`provenance.py:359`, `run.py:379`, `dataset.py:163`, `user.py:113`). Under that
reading, the boundary question ("if create-dataset is MOU's it should not be
added as a *hasPart* of the main activity") is concrete: it decides whether the
dataset-creation run appears in the main activity's `hasPart` list. Worth
confirming that reading with whoever took the notes, since T8 depends on it.
Speaker names ("Mou/MOU", "Matúš", "Irká", "Mirko") are also flagged uncertain in
the summary — verify before naming owners in T12.

---

## Tier 1 — small, self-contained

### T1. Add `prov` + graphviz to the environment — ✅ DONE (graphviz: documented, not installable here)
Implemented:
- `pyproject.toml`: new `[dependency-groups] prov = ["prov[dot]>=3.2"]`. Kept out
  of runtime deps (only the script renders graphs). `uv lock` refreshed —
  note the lock also picked up `pytest`/`pluggy`/`iniconfig` that were in the
  dev group but missing from the lockfile.
- **`prov[dot]`, not `prov`**: plain `prov` fails at `import prov.dot`
  (`pydot` is an optional extra). Caught by actually running it.
- graphviz `dot`: **cannot be installed in this sandbox** (no sudo, no apt
  index; linuxbrew exists but out of scope). `export_provenance_png()` now
  degrades to writing the Graphviz source (`prov-<name>.dot`) with a warning,
  so every environment produces an inspectable artifact and the PNG is one
  `dot -Tpng` away wherever `dot` exists. Requirement documented in pyproject
  comment.
- `provenance.py`: `chmod +x` (has a shebang; was tripping ruff EXE001).
  **Not yet `git add`ed** — left for you to commit alongside your own work.
- verified: `--self-check` passes; a stub-driven e2e build writes `.provn` +
  `.dot` containing the CPM nodes. **Real PNG still unverified** — needs a
  machine with `dot` AND network to the tracking server (unreachable here).

### T2. ~~Single source of truth in the assembler~~ — OUT OF SCOPE
Dropped by decision. Consequence: every structural change lands in
`build_provenance()` **and** `build_prov_document()`; run `--self-check` after each.
(Partially softened in practice: T3's main-activity *attributes* and id now go
through one shared helper used by both paths — attributes are single-sourced,
the walks remain duplicated.)

### T2a. Reconcile the main-activity identifier — ✅ DONE (script half; mlkit half pending)
Implemented in `provenance.py`:
- one construction site: `main_activity_id(root_id)` → `blank:Run_<id[:10]>`
  (kept the script's existing 10-char scheme; it was the odd builder using 8)
- `--self-check` asserts the id shape and greps the output for the stale
  `TrainingRun_` spelling
- **still open in mlkit** (`run.py:407` `TrainingRun_<8>`,
  `log_provenance.py:295` `Run_<8>`): invisible in the graph (finding B), but
  wrong per D2 if anyone feeds a per-run `prov.json` to the Java tool.
  Fold into T5's decision on whether per-run builders keep a main activity.

### T3. Main-activity placeholders — ✅ DONE (values), ⚠️ names now in doubt (finding D)
Implemented in `provenance.py`:
- `CPM_PLACEHOLDERS` module dict (both attributes, `TODO:` sentinel values) +
  one `# TODO(prov-next-iteration)` marker → greppable, removable in one edit.
- `cpm:referencedMetaspectVersion` added to the main activity on **both** paths.
- `cpm:referencedMetaBundleId` uses the placeholder only as a *fallback*: where
  the bundle's own `meta:<run_id>` entity exists it is used, since inventing a
  fake id there would be strictly worse than the real one. The placeholder
  remains for the eventual MOU/external metabundle case.
- shared builder `main_activity_props(meta_id, has_parts)` feeds both paths, so
  the attributes cannot drift.
- verified: `--self-check` asserts both keys exist on both the JSON and the
  diagram element, and that the JSON one resolves to the real meta id.
- superseded: the `CPM_CONNECTORS` forward/backward placeholder added here as T5
  scaffolding was replaced in T5 by the published `cpm:currentConnector` /
  `cpm:forwardConnector` terms (finding C). The two placeholder *values* above
  remain, and finding D now questions their *names* too.

### T4. Make the diagram contain what the JSON contains — ✅ DONE (PNG pending on `dot`)
Implemented in `provenance.py::build_prov_document`:
- `blank:` / `meta:` / `cpm:` / `xsd:` added to `NAMESPACE_URIS` — required,
  `prov` raises on unregistered prefixes.
- attribute translator `_attrs_for_doc`/`_add_attrs`: converts the JSON-LD
  `[{"type":"prov:QUALIFIED_NAME","$":id}]` form into prov-native values (prov
  rejects the list-of-dicts form as unhashable). Returns (key, value) **pairs**,
  not a dict, so multi-valued `dct:hasPart` survives.
- diagram now carries: `meta:<root>` BundleMetadata entity, the
  `cpm:mainActivity` activity (via T3's shared helper), and one
  `prov:Annotation` entity per run per payload (params, environment), values
  truncated at `_MAX_ATTR_CHARS = 120` because `prov.dot` renders every
  attribute into the image.
- **answer for the team:** the JSON was correct; the diagram path simply never
  emitted these nodes. "Wrong picture" hypothesis not needed — reproduced in code.
- two divergences found & fixed while implementing, worth knowing:
  1. annotation content key is `prov:annotation` (JSON) — an earlier draft
    wrote `prov:comment`; now aligned.
  2. **`prov` 3.2.2 has no `qualifiedAssociation` record type** (checked
     `prov.model.records`). The JSON emits that section; the diagram cannot.
    The diagram encodes the subject as a `prov:annotatedEntity` attribute
    instead, so the *node* matches and only the *edge* waits for T5.
- verified via stub: main activity + meta + both annotation nodes present in the
  `ProvDocument` and in the Graphviz source (`mainActivity` nodes count = 2).
  Real `.png` still unverified here (no `dot` binary, no server).

---

## Tier 2 — structural graph work

### T5. Connect the main activity + make it chain-level `[REPO]` — ✅ DONE (assembler) / mlkit half open
Implemented in `provenance.py`, on **both** builder paths:
- **Real relation records, not just properties.** The main activity now has one
  inbound and one outbound edge, exactly as the backbone template prescribes:
  `mainActivity used currentConnector` (chain input boundary) and
  `mainActivity wasGeneratedBy forwardConnector` (chain output). In PROV-N this
  renders as `used(blank:Run_aaaaaaaaaa, gen:current_…)` and
  `wasGeneratedBy(gen:forward_…, blank:Run_aaaaaaaaaa)` — verified in a stub run.
- **`dct:hasPart` now spans the whole chain** (`[f"gen:run_{rid}" for rid in
  chain]`), replacing the root-only list. This is D1: the main activity is what
  the team is responsible for, so the end of the chain can be referenced from it.
- **Connectors are named with the published CPM terms**, not the earlier
  `cpm:TODO-forwardConnector` guesses. `CPM_CONNECTORS` (the TODO scaffolding) is
  replaced by `CPM_CURRENT_CONNECTOR` / `CPM_FORWARD_CONNECTOR` constants, with
  the namespace and template URLs cited at the definition so the spelling is
  checkable. See finding C.
- **`specializationOf` from each connector to the domain entity it stands for**
  (`currentConnector → gen:input_…`, `forwardConnector → the output entity`).
  This is what the template means by "inputs and outputs … are related using the
  PROV specialisation relation". The `prov` package supports it natively
  (`ProvBundle.specializationOf(specific, general)`, record class
  `ProvSpecialization`); the JSON path gets a new `specializationOf` section using
  the grammar's `prov:specificEntity` / `prov:generalEntity` fields.
- **Chain input boundary** = `chain_input_uris()`: artifacts the chain consumes
  but does not produce (`up_id not in chain`). Deduplicated, first-seen order, so
  one connector per real external input rather than one per mention.
- **No `backwardConnector` / `receiptActivity`** — deliberately absent, see
  finding E. `--self-check` asserts their absence so their appearance is a
  decision, not an accident.
- Verified by `--self-check` on both paths: the generation and usage records are
  found by inspecting `formal_attributes` (prov exposes no stable per-endpoint
  property), and `hasPart` is compared as a *set* because prov stores
  multi-valued attributes in a Python `set`, so the diagram path has no defined
  order while the JSON path does.

**Still open (mlkit half, deliberately not touched):** `run.py:366-381`,
`log_provenance.py:383`, `user.py:108`, `dataset.py:158` each still emit their own
per-run main activity with no relation endpoints. Per D1 that is now redundant —
one main activity belongs over the chain, which is the assembler's job
(recommendation unchanged: drop it from the per-run builders). Not changed here
because finding B means those documents never reach the graph, so editing them
would look like progress without changing anything anyone sees.

### T6. Model entity + forward connector `[REPO]` — ✅ DONE (assembler) / mlkit half open
Implemented in `provenance.py`, both paths:
- **The chain end's output is model-typed when that run trained a model**:
  `gen:model_<run_id>` instead of `gen:output_<run_id>`, typed
  `prov:Entity` + `schema:CreativeWork`, carrying `schema:url` of the model
  artifact and a `dct:description` that records *how* the artifact was
  identified. `prov:Entity` is included so a consumer resolving only PROV types
  still sees an entity; `schema:CreativeWork` is the declared type. Deliberately
  **not** `mlflow.models.Model` or a flavor name — no term here implies a
  serialization format the repo does not pin down. A separate `gen:model_` id
  (rather than retyping `gen:output_`) is what lets MOU matching and a reader of
  the graph tell the trained model apart from dataset outputs.
- **No duplicate node**: the model-typed entity *is* that run's output entity, so
  the graph does not grow a second node for one artifact. Upstream outputs stay
  `sosa:Sample`.
- **Artifact narrowing** (`model_artifact_uri`): walks the run's artifacts and
  returns the most specific model-looking path, so the entity points at
  `…/artifacts/model/MLmodel` rather than the whole artifact directory. Falls
  back to `run.info.artifact_uri` when the listing fails or nothing matches, and
  says so in `dct:description` — a wrong guess is visible in the document.
  `mlmodel` sorts first because MLflow's own python_function flavor marker is the
  strongest evidence of a real logged model.
- **Which run is "the model producer"**: only the chain end is considered, and
  `is_model_run()` accepts either a model-looking *input* (fine-tuning /
  evaluation / inference sit downstream of training) or a training-looking run
  *name*, so a training run that died before logging a model still yields a model
  entity. Cost of a wrong guess is a mislabelled entity, not a broken graph.
- **Forward connector** = `cpm:forwardConnector`, `wasGeneratedBy` the main
  activity, `specializationOf` the model entity, `schema:url` the same artifact
  URI. **Not** "Specialized Forward Connector" — the type is asserted explicitly
  so nothing can read it as that other object. `--self-check` pins the type.
- Closes the meeting action item "create the entity representing the trained
  model (output) and specialize/link the forward connector to it".

**Still open (mlkit half):** `run.py` still emits no output entity at all
(`wasGeneratedBy` only for `meta:<run_id>`, `run.py:360-363`), so per-run
documents still dead-end at training. Same reasoning as T5 for not touching it
blind; finding B applies.

### T7. Dataset entity + explicit `used` `[REPO + DECISION]` — ~2 h + a decision
Partly present: `_add_input_entities` already creates a dataset entity **and** a
real `used` edge (`run.py:97-155`). The gap is identity — the fallback is
`gen:dataset_<first-8-of-training-run-id>` (`run.py:144`), which names the
*training run*, not the dataset.
**(M) Two open questions now gate the design:**
- **Two inputs, not one** (D5): index + real images. The images are an on-disk
  path, which `URI_RE` cannot detect, so they are absent from the graph entirely.
  Needs an explicit input for a local path — new capability, not a rename.
- **One entity or two?** (single dataset collection vs index + on-disk files) —
  meeting left it flexible/unstandardised. Needs a call before implementing;
  recorded as **T13a**.
- `dataset_run_id` still comes from `lookup_dataset_run()` = *newest run in
  `Dataset_Registry`, globally* (`dataset.py:208`). An explicitly wired "uses
  this dataset" edge whose value is a guess is worse than none — pin it first.

### T8. Preprocessing subgraph VSI → index `[REPO + BLOCKED-on-data]` — ~1-2 d
New modelling: VSI directory, tile cutting/tile metadata, slides metadata,
resulting dataset, train-dataset connector → VSI index, dataset derived from
index. `log_provenance` yields only run → inputs → outputs
(`log_provenance.py:267-414`); intermediate entities need a new builder.
- **(M) Cannot be verified against real data** — no access to the MOU cohort (D7).
  Build against a dummy VSI tree; expect the shape, not the content, to be right.
- **(M) Scope depends on T13** — if create-dataset is MOU's, its node must not be
  a `hasPart` of our main activity (reading "Hespart" as `dct:hasPart`).
- depends on T9 for types/predicates
- **gap that truncates the chain:** `URI_RE` matches only `mlflow-artifacts:/`.
  Local-path inputs leave no edge. `log_provenance.py:67` accepts `runs:/` and
  `models:/` too; the assembler regex does not. Fixing this is what lets the
  chain actually reach the VSI directory.

---

## Tier 3 — verification

### T9. Reference documents `[PARTLY RESOLVED]` — was listed as prerequisite for T5, T6, T8
**What turned out to be public:** the CPM namespace and the backbone template are
published and reachable, and T5/T6 were implemented against them (see findings C
and E). The old framing — "any `cpm:` predicate invented before T9 is a guess" —
was wrong for the connector terms; they were findable all along.

**What is still genuinely missing:**
- **the three non-standard terms** (`BundleMetadata`, `referencedMetaBundleId`,
  `referencedMetaspectVersion`): which namespace version defines them, if any —
  this is now the highest-value question in the whole list, because it decides
  whether T3's placeholders are a value to fill in or names to replace (finding D)
- the ISO standard text and Matej Gala's figure, for reviewing the *shape* of the
  graph against a canonical drawing rather than against our reading of the template
- **T8 is the remaining consumer**: its per-node types (VSI directory, tile
  metadata, slides metadata) are not terms the CPM namespace defines, so T8 still
  needs either the reference material or a decision to use local vocabulary
- **(M)** still needed *before next week's meeting* if the graph is to be reviewed
  against the reference, so raise the request early.

### T10. MLflow & hash pipeline `[REPO + process]` — ~4 h
The workflow is confirmable, but the premise is missing: **no hashing exists in
the repo** — no `hashlib`; `verify_manifest` compares **file sizes only**
(`dataset.py:359-367`) despite the header "Hash-based registration (preferred)".
A file edited to identical length still reports `VERIFIED`. **(M) Under D4 this
is the mechanism that binds the graph to real data**, so it is load-bearing,
not hygiene.
- add sha256 in `register_dataset` (`dataset.py:445-453`), compare in
  `verify_manifest`; store as **artifact, not tag** — `file_sizes`/`file_mtimes`
  are already JSON-in-a-tag (`dataset.py:479-480`) and will hit MLflow's tag
  length cap on real cohorts
- `provenance.py:217-226` already sha256s environment artifacts — different
  purpose, but the helper may be shareable
- **(M) D3 follow-up:** registration runs emit their own `prov.json`
  (`dataset.py:503`, `user.py:216`). Is registration "the generation process"
  that must stay out, or legitimate dataset-history content? Decide.

### T11. Reconcile JSON vs diagram → covered by T4 `[REPO]`
Line item kept so it isn't dropped; the work is T4, and it is already diagnosed.

### T10b. Can the graph answer "on which files was this model trained?" `[REPO]`
**(M)** The meeting left this open and case-by-case. It is the practical test of
all of the above, and today the answer is **no, not from provenance alone**:
file-level identity lives in MLflow tags/artifacts (per-file sizes only, no
hashes), and the real image paths are invisible per D5. Add this as an explicit
acceptance check once T6/T7/T8 land — if it fails, the graph is decorative.

---

## Tier 4 — decisions and coordination

### T13a. One dataset entity or two (index + on-disk files)? `[DECISION]` **(M)**
Considered flexible at the meeting. Blocks T7's implementation shape.

### T13. System boundary: is "create dataset" ours or MOU's? `[NOT CODE]`
Decides whether dataset creation is a `hasPart` of the main activity and how much
of T8 is ours. Answer **before** T8 starts, not during.

### T12. Ownership per action item `[NOT CODE]`
Especially MOU coordination. **(M)** Speaker attribution in the summary is
explicitly inferred — confirm names before assigning.

### T14. MOU identification query `[NOT CODE]` — deferred by decision, still ask
**(M)** D6: linkage is deferred, chain legitimately starts at our end. But the
question still has to go out, and per T2a it must ask in terms of *our* id scheme.
Ask specifically: what identifier will MOU mint for their dataset entity, and
what will their **backward connector** point at — our `cpm:forwardConnector`
(now emitted, T6) is the node their side specialises from, so the merge only
works if their backward connector targets our forward connector's exact id.
Also worth asking whether they expect us to emit `receiptActivity` (finding E):
if the merge point is meant to carry it, somebody has to, and it is not us.
Once answered, wiring is small: `get_prov_prefixes()` override / `PROV_BASE_URI`
(`common.py:33-50`) + entity id construction.

---

## Suggested order

~~T1~~ ~~T2a~~ ~~T3~~ ~~T4~~ ~~T5~~ ~~T6~~ done (assembler) → **next: T7** (needs
the T13a decision) → T10. T13 before T8. T10b as final acceptance check.
T12/T14 from day one. Two cheap high-value asks, both now: the namespace-version
question from finding D, and whether the per-run main activities in mlkit should
be dropped (the mlkit half of T5/T6 is mechanical once that is decided).

## Verification state (after T1-T6)

- `python provenance.py --self-check` → **passes**. Now also asserts the T5/T6
  backbone on both paths: the main activity's `used` / `wasGeneratedBy` records
  exist with the right endpoints, connector types are the published CPM terms,
  `specializationOf` points the right direction, the model entity exists with its
  type and artifact URI, `hasPart` spans all three stub runs, and
  `backwardConnector` / `receiptActivity` are absent.
- `ruff check` + `ruff format --check` on `provenance.py` → **clean**.
- `mypy --strict` → **31 errors, none in code added here.** Verified with an AST
  pass mapping each error to its enclosing function: no error falls inside any
  function introduced by T5/T6, and the new code is fully annotated
  (`dict[str, dict[str, Any]]`, `dict[str, dict[str, str]]`). The count moved
  27 → 31 because `build_provenance()`'s local section dicts now sit inside a
  longer function body at lines my earlier line-range check attributed to older
  code — same pre-existing bare-`dict` pattern, one of them (`specialization_of`)
  is now annotated. All 31 are the same class of pre-existing debt: bare `dict`
  generics, untyped `Probe` methods, `SimpleNamespace` where `Probe` is expected.
- **Verified by running it**, on a stub chain shaped like the real thing (train
  run whose artifacts contain `model/MLmodel`): model entity resolves to
  `…/artifacts/model/MLmodel`, forward connector carries the same URI, PROV-N
  contains `used(blank:Run_…, gen:current_…)`,
  `wasGeneratedBy(gen:forward_…, blank:Run_…)` and both `specializationOf` lines,
  and the DOT source contains the mainActivity / forwardConnector /
  currentConnector / model nodes.
- **Still not verified anywhere:** a real `.png` and a real chain against the
  server. Blocked twice over in this sandbox — no `dot` binary (the export
  degrades to writing `.dot` source), and the tracking server
  (`mlflow-jiribuchta.dyn.cloud.trusted.e-infra.cz`) is unreachable from here.
- **Judgement calls to review**, since neither is verifiable from the repo:
  `schema:CreativeWork` as the model's type, and `is_model_run()`'s heuristic for
  deciding which run produced the model.

## Files changed by T1-T6

| file | change |
| --- | --- |
| `provenance.py` | `NAMESPACE_URIS` +4 prefixes; `CPM_PLACEHOLDERS`; `CPM_CURRENT_CONNECTOR` / `CPM_FORWARD_CONNECTOR` (replacing the `CPM_CONNECTORS` TODO scaffolding), `MODEL_TYPES`, `MODEL_ARTIFACT_HINTS`, `MODEL_RUN_NAME_HINTS`; `main_activity_id()`, `main_activity_props()`, `model_entity_id()`, `forward_connector_id()`, `current_connector_id()`, `input_entity_id()`, `chain_input_uris()`, `is_model_run()`, `model_artifact_uri()`, `run_base_uri()`, `resolve_output()`, `output_id()`, `output_props()`; `_qn_for()`, `_attrs_for_doc()`, `_add_attrs()`, `_MAX_ATTR_CHARS`; both builders emit the backbone (connectors, `specializationOf`, model entity, chain-wide `hasPart`); `build_prov_document` takes `include_environment` and emits meta + main activity + annotations; `export_provenance_png` DOT fallback; `self_check` extended with a non-resolving 4th stub run; docstring/style fixes; `+x` |
| `pyproject.toml` | `[dependency-groups] prov = ["prov[dot]>=3.2"]` + graphviz note |
| `uv.lock` | refreshed (prov, pydot, pyparsing, networkx, pytest, pluggy, iniconfig) |
| `PROV_TODO.md` | this file |

Not committed. `provenance.py` is still untracked — decide its home (`tools/`?)
before `git add`.

## Pre-existing issues found while scoping (not in the task list)

- **The diagram declares the same agent once per run.** `build_prov_document()`
  calls `bndl.agent(f"gen:user_{user}", …)` inside the per-run loop, so a chain
  whose runs share an author emits that person twice: verified on a two-run stub,
  two `ProvAgent` records both identified `gen:user_u1`. The JSON path is correct
  because it uses `agent.setdefault(...)`. Fix is one-node-per-thing (track
  created agents in a local dict, mirroring `setdefault`). Not done in T5/T6 —
  unrelated to the backbone, and the file has no test beyond `--self-check`.
  Cheap and visible in the PNG; worth a yes/no before touching it.
- **Run activities carry no `prov:type` in the diagram path.** The JSON writes
  `schema:Action` (`build_provenance`), `build_prov_document` never sets a type on
  `gen:run_<id>`, and neither does it type `gen:input_…` as `sosa:Sample`. So T4's
  parity holds for *nodes and relations* but not for every *attribute*; T5/T6
  added type parity for exactly the nodes they introduced (model, connectors) and
  left the older ones as they were.
- README documents `log_dataset.py` / `log_split.py` (`README.md:370-371`),
  deleted by `a391b9f`; claims "Container ID, image name + hash"
  (`README.md:135`) though detection is now one env var; the `examples/` tree the
  quick start tells you to run is gitignored and absent.
- `.github/workflows/pytest.yml` runs on master/PRs with `testpaths = ["tests"]`
  but no `tests/` directory exists → that workflow fails. `provenance.py`'s
  `--self-check` is the only test-like thing in the repo and CI never calls it.
- `positive_label` counts the *negative* class (`log_provenance.py:190`).
- Callback ordering: `if has_env or has_verify: gather() else: fallback()`
  (`lightning/callbacks/provenance.py:578`) — if `ProvenanceCallback` precedes
  `EnvironmentCallback`, siblings read `None`, no fallback runs, and the PROV doc
  silently lacks environment data.
