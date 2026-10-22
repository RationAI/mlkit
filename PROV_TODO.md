# PROV graph — TODO (post-review rewrite)

Rewritten after the external review of the generated graph
(`cpm_review_7c06254109.docx`, review of `provenance_7c06254109fb4cb298cf352bcd0ebc2c.prov.json`,
reviewed against `CPM_knowledge_base.md` rev. 9 (KB) + W3C PROV-DM / PROV-JSON /
PROV-CONSTRAINTS, dated 2026-10-06). Each item below carries the review's finding
ID (`R…` / `O…`) so the docx and this file can be cross-read. Verbatim search
strings from the JSON (given in the docx per finding) are the anchors.

Previous version of this file (with the T1–T6 implementation log and the meeting
transcription notes) is preserved in `PROV_TODO.pre-review.md.bak`.

**Review numbering is non-contiguous**: R6–R9, R13, R16 do not exist in the docx.
R6 is referenced from R15 (slides not modelled) and folds into the preprocessing
work (old T8). Do not renumber the review's IDs.

Tags: `[REPO]` code change here · `[DECISION]` needs a human call first ·
`[BLOCKED]` needs an input we lack · `[NOT CODE]` org/process.
Severity per review: **Critical** = breaks PROV validity / CPM conformance /
traversal · **Major** = wrong or missing semantics · **Minor** = hygiene.

## Where the review leaves us

The generated document today: **does not parse** as PROV-JSON (R10, R11), is
**invalid** under PROV-CONSTRAINTS (R12), and is **not CPM-conformant** (R1–R4).
The mechanical-validation section of the review confirms: after stripping
`qualifiedAssociation` the doc parses, but all 7 `wasDerivedFrom` records come
out with *both* arguments `None` — the chain is disconnected for every standard
consumer. Temporal ordering is consistent for all 19 usages (the one thing that
survives).

Two of our earlier *deliberate* decisions are contradicted by the review and
must be revisited first (see D-a, D-b below) — several P0 items depend on them.

The review also resolves parts of the old "finding D" question:
- the CPM namespace is `https://www.commonprovenancemodel.org/cpm#` (not
  `cpm-namespace-v1-0/`, and not answered by the published namespace page alone
  — the KB is the normative source here) → R1;
- the correct spelling of the mandatory main-activity attribute is
  `cpm:referencedMetaBundleSpecV`, value format `ISO-[DOC_TYPE]-[PROJECT-CODE]-[PART]:[YEAR]` → R4;
- still open: `cpm:BundleMetadata` / `cpm:organization` on the meta entity are
  not KB terms either → R22.

## Decisions needed before/while fixing (D-a … D-e)

**D-a (from R10) — abandon the old `prov:entity`/`prov:derivation` spelling?**
Old finding F kept the non-PROV-JSON field names deliberately, to match a Java
`prov_mlflow` consumer. The review shows the cost is total: no standard consumer
can read any derivation. Unless someone demonstrates that Java tool actually
parses this exact dialect, switch to the grammar's `prov:usedEntity` /
`prov:generatedEntity` **everywhere** (both builders + `--self-check`
assertions). Default recommendation: **fix it**; the "contract" was never
verified and the review proves the current shape is lossy.

**D-b (from R2) — `specializationOf` direction.** T5 implemented
`specializationOf(connector → domain entity)` citing the published backbone
template, and `--self-check` pins that direction. Review R2 (KB B.3, normative)
says CPM requires `specializationOf(domain entity, connector)` — the domain
entity is the *specific* one. Recommendation: trust the KB, flip the direction,
and update the self-check that currently asserts the wrong thing. If the
template figure genuinely reads the other way, that is a KB-vs-template question
to raise with the reviewer, not a reason to keep both.

**D-c (from R11) — where do params/environment payloads live?**
`qualifiedAssociation` + `prov:Annotation`/`prov:annotation`/`prov:annotatedEntity`
are not PROV terms (we minted terms in the `prov:` namespace — worst place to
invent vocabulary). Options: (a) plain attributes on the activity under our own
namespace (e.g. `mlkit:params`, `mlkit:environment`), (b) entities `used` by the
activity. Smallest faithful change is (a). Needs a call because it changes the
consumer-facing shape.

**D-d (from R14) — one entity or two per artefact?** Option (a)
`wasGeneratedBy(file, run)` directly, drop the folder entity (simplest, review's
lean); option (b) folder as `prov:Collection` + `hadMember(folder, file)` if the
folder itself matters. Same family as the old T13a question.

**D-e (from R3/R4) — current agent identity + spec-version value.** The
organisation is derivable (`RationAI`), but `cpm:contactIdPid` is a placeholder
and `cpm:referencedMetaBundleSpecV` needs a real `ISO-[DOC_TYPE]-[PROJECT-CODE]-[PART]:[YEAR]`
value we do not have in the repo. Who owns these two strings?

## P0 — make the document parse, be valid, and conform (all Critical)

**Status (2026-10-14): all P0 done.** R1, R12, R17 (2026-10-13, `output4/`) +
R10, R11, R2, R3, R4 (2026-10-14, `output5/`, snapshot
`reference/post-P0-complete/`). Decisions taken as expert, per
`cpm_review_7c06254109.docx`: **D-a → fix** (PROV-JSON grammar keys, verified
against the prov package: `generatedEntity` = direct entity, `usedEntity` =
indirect source); **D-b → KB direction** (domain entity is
`specificEntity`, connector is `generalEntity`); **D-c → option (a)**
(`mlkit:params` / `mlkit:environment` as custom activity attributes on both
paths — no `qualifiedAssociation`, no annotation entities); **D-e →
placeholders** (`gen:org_RationAI` typed `cpm:currentAgent` +
`prov:Organization`, `cpm:contactIdPid: "TODO:contactIdPid"`, and the
`referencedMetaBundleSpecV` rename). R5 stays open (P1). The P0 acceptance
gate is wired into `--self-check`: the prov-document serializes to PROV-JSON
and PROV-N and both deserialize cleanly with every derivation endpoint and
association agent (3 runs + org agent) intact, and the self-check asserts the
R10 grammar keys, the R2 direction, the R3 org agent and the R4 spelling.

Ordered so each step unblocks testing the next. Every item touches
`build_provenance()` (JSON path) **and** `build_prov_document()` (diagram path),
per old T2's "no single source of truth" decision — run `--self-check` after each.

- **R10 [DECISION D-a → REPO] `wasDerivedFrom` uses non-PROV-JSON keys. ✅ DONE (2026-10-14).**
  Decision: **fix** (option A — the document must conform to PROV-JSON as the
  review's acceptance requires). Both chain links and the forward←current
  connector now use `prov:generatedEntity` / `prov:usedEntity`, kept in lockstep
  with the prov path: `bndl.wasDerivedFrom(inp, out)` emits
  `generatedEntity` = first argument (the direct entity, e.g. the input
  artifact), `usedEntity` = second (the indirect source, e.g. the upstream
  output). Self-check asserts the grammar keys and the round-trip gate pins
  both endpoints after deserialize.

- **R11 [DECISION D-c → REPO] `qualifiedAssociation` + `prov:Annotation` are not
  PROV terms. ✅ DONE (2026-10-14).** Decision: **option (a)** — re-home the
  payloads as custom `mlkit:params` / `mlkit:environment` attributes on the
  run activities (full payloads in the JSON path, truncated to 120 chars on
  the diagram path because `prov.dot` renders every attribute). The
  `qualifiedAssociation` section, the annotation entities, and the
  `wasInfluencedBy` edges are gone from both paths; the `mlkit` namespace
  (`http://rationai.org/prov/mlkit#`) is registered in `PREFIX` +
  `NAMESPACE_URIS`. Self-check asserts the annotations are absent and the
  attributes present on both paths.

- **R12 [REPO] `wasAttributedTo` with activities as subject. ✅ DONE (2026-10-13).**
  8 records where the "entity" is a `gen:run_*` activity → entity/activity
  disjointness violated, document invalid. Emitted at `provenance.py:548-556`
  (JSON) and `provenance.py:851` (diagram). Review: redundant with
  `wasAssociatedWith` — **dropped the records entirely** from both paths;
  `--self-check` now asserts their absence so they cannot drift back in.

- **R1 [REPO] cpm namespace. ✅ DONE (2026-10-13).** `cpm-namespace-v1-0/` →
  `https://www.commonprovenancemodel.org/cpm#` in `PREFIX` and `NAMESPACE_URIS`
  (`provenance.py:57,79`) **and**
  `rationai/mlkit/provenance/common.py:23` (the per-run builders share the
  wrong namespace). Verified in `output4/`.

- **R2 [DECISION D-b → REPO] `specializationOf` reversed. ✅ DONE (2026-10-14).**
  Decision: follow the KB (B.3) — the **domain entity is the
  `specificEntity`**, the connector the `generalEntity`. Flipped on both
  paths (JSON `specializationOf` section and the prov
  `bndl.specializationOf` argument order); self-check now pins the
  domain→connector direction so it cannot regress. Note: in the current
  live chain there is no external input (the chain is closed inside MLflow),
  so only the forward-side specialization appears; the input side is
  exercised by the self-check's stub chain.

- **R3 [DECISION D-e → REPO] no current agent. ✅ DONE (2026-10-14).**
  Decision: placeholders (the real `contactIdPid` / `referencedMetaBundle*`
  values come from the CPM side later). `gen:org_RationAI` is declared with
  `prov:type: [cpm:currentAgent, prov:Organization]`,
  `schema:name: RationAI`, `cpm:contactIdPid: TODO:contactIdPid`, and is
  `wasAssociatedWith` the **main activity** (not the individual runs) on both
  paths. Multi-valued `prov:type` goes through `_add_attrs` — the prov
  record constructor stores a list value as one unhashable element.

- **R4 [DECISION D-e → REPO] `cpm:referencedMetaBundleSpecV` missing. ✅ DONE (2026-10-14).**
  Decision: placeholder, same value style as the existing
  `referencedMetaBundleId`. Renamed in `CPM_PLACEHOLDERS` +
  `main_activity_props`; the self-check asserts the misspelling is absent.
  The per-run builders (`run.py`, `dataset.py`, `user.py`,
  `log_provenance.py`) turned out to emit **neither** spelling — only
  `provenance.py` does, so no further rename was needed. Real value format
  when it becomes available: `ISO-[DOC_TYPE]-[PROJECT-CODE]-[PART]:[YEAR]`.

**Acceptance for P0** (the review's own method, repeatable): the document must
`ProvDocument.deserialize(format='json')` cleanly with **no** keys removed, and
all derivations must survive with both endpoints intact. Wire this into
`--self-check` so parseability is regression-tested, not just shape.

## P1 — Major (wrong or missing semantics)

- **R5 [REPO] `cpm:referencedMetaBundleId` must be a full IRI typed `xsd:anyURI`,
  never a qualified name.** Currently `{"type": "prov:QUALIFIED_NAME", "$": "meta:…"}`
  (`provenance.py:155-156`, `_qn(meta_id)`; same pattern in mlkit builders,
  e.g. `log_provenance.py:385`). Target shape per KB B.6:
  `{"$": "http://…/api/v1/documents/meta/7c06…", "type": "xsd:anyURI"}`.

- **R14 [DECISION D-d → REPO] two entities per artefact.** Each run emits an
  `output` (artifacts folder) plus per-file `input` entities linked only by the
  (currently broken) derivations; no activity generates the file entities.
  Anchors: `gen:output_ef60…` vs `gen:input_mlflow_artifacts_1_…`.

- **R15 [REPO + BLOCKED on producer info] `used` is over- and under-approximated.**
  Over: the training run records 6 inputs — every MLflow URI in the config —
  though the data module reads only the resolved `embeddings/train` paths.
  Under: Dataset Creation and Filter Tiles have **no** `used` at all; the slides
  they consume are unmodelled (absent finding "R6" → old T8 preprocessing
  subgraph). The over-approximation is the assembler trusting config keys;
  verify with the producer what was actually read.

## P2 — Minor (hygiene)

- **R17 [REPO] ✅ DONE (2026-10-13).** `gen:user_jiribuchta` now typed
  `prov:Person` + `schema:Person` on both paths (`provenance.py:549` / agent
  block in `build_prov_document`). Name is personal data, stays as given.
- **R18 [REPO]** `"gen": "gen/"` is not an absolute IRI → identifiers are not
  globally unique (`proven.py PREFIX`/`NAMESPACE_URIS`; also `common.py`).
- **R19 [NOT CODE, accepted for now]** bundle/meta-bundle ids on
  `http://localhost:8083/…` — fine for a test, not for a shareable FPC.
- **R20 [REPO]** main activity lives in the blank namespace
  (`blank:Run_…`, `main_activity_id`, `provenance.py:165-168`) — ProvToolbox
  blank-node territory; a main activity needs a stable, named identifier.
  Supersedes the T2a resolution: the reconciled id also has to leave `blank:`.
- **R21 [REPO]** real-world ids are carried as `schema:identifier` /
  `schema:url`; CPM's mechanism is `cpm:externalId` + `cpm:externalIdType`
  (e.g. "MLflow run ID", "MLflow artifact URI") on domain entities and
  activities. Also: our PROV identifiers are built *from* MLflow run IDs/artifact
  URIs, against the project convention (KB A.3).
- **R22 [REPO + question to KB owner]** the `meta:` entity inside the FPC
  (`cpm:BundleMetadata`, `cpm:organization`, `gen:input_uris`,
  `provenance.py:645-651` and diagram path ~930) uses terms the KB does not
  define (old finding D, still open for these two). Meta-component content
  belongs to the meta-component (KB B.1); `gen:input_uris` duplicates the `used`
  relations and should go. **Ask:** where does bundle-level metadata live if not
  in the FPC?
- **R23 [REPO, optional]** `cpm:hashValue` + `cpm:hashAlg` on the model entity —
  optional on forward connectors; the assembler already sha256s environment
  artifacts (`provenance.py` env hashing), so the helper exists. Natural
  companion to the old T10 (hash the data, not just describe it).

## Observations on recorded content (docx §4 — not graph defects)

Tracked so they reach the right owner; most are `[NOT CODE]` or producer
questions.

- **O1** `tiling_filtered.train` denotes *different artefacts* in two runs
  (embeddings vs filtered tiles) — config-key collision, worth flagging to
  whoever owns dataset naming.
- **O2 [REPO, mlkit]** training config recorded with unresolved Hydra
  interpolations (`cfg_seed`: `"${random_seed:}"` …) — **the random seed actually
  used is not in the graph.** Fix belongs in the mlkit logging path
  (`log_provenance.py` param capture): resolve interpolations before recording.
- **O3** Filter Tiles captured with `"prov:annotation": ["{}"]` and no
  environment, though its output feeds the embeddings run.
- **O4** embeddings run (virchow2) records `gpu_count: "0"` +
  `cuda_runtime: "n/a"` alongside `cfg_concurrency: 512` — possibly driver
  process resources, not workers. Check with the producer.
- **O5** all runs share one `container_image` tag but lock-file hashes differ
  (e.g. `pyproject.toml` `8ae0c3a5…` vs `2fa1e446…`) — the tag alone does not
  identify the environment; the hashes do. Related: R23/T10.
- **O6** split sizes 156+31+42=229 consistent; but both test fractions are 0.15
  with different row counts (31 vs 42) — presumably case-level grouping,
  unverified.
- **O7** temporal order consistent for all 19 usages — no action.

## Carried over from the previous TODO (unchanged in substance)

- **T7 dataset entity + explicit `used`** — still gated on D5 (index + real
  images as two inputs; local paths invisible to `URI_RE`) and on D-d above
  (one entity or two). `dataset_run_id` still guessed via
  `lookup_dataset_run()` (newest globally, `dataset.py:208`) — pin before wiring.
- **T8 preprocessing subgraph VSI → index** — still blocked on real data (D7:
  build/verify against synthetic VSI tree) and on T13 (boundary). R15's missing
  `used` for Dataset Creation / Filter Tiles lands here, as does the absent
  slides modelling.
- **T10 hashing** — `verify_manifest` compares file sizes only; add sha256,
  store as artifact not tag. Now also carries R23 (model hash) and D4's
  "bind to real data".
- **T10b acceptance check** — "on which files was this model trained?" — still
  the practical test; add it after P0+P1+T7/T8 land.
- **T12 / T14 coordination** — ownership per action item; MOU identification
  query (what id will MOU mint, what will their backward connector specialise
  from — our `cpm:forwardConnector`). Note for T14: after R2's flip, the merge
  direction statement changes wording but not substance.
  **CORRECTION (2026-10-13): the direction in this carried-over text is wrong
  and contradicts the 2026-09-21 meeting.** MOU is the *sender* of the dataset,
  so per the published backbone template (verified
  <https://www.commonprovenancemodel.org/cpm-backbone-template-v1/>) their side
  emits a **forward** connector and **our** chain starts with a
  **backward** connector + `receiptActivity` (the receiving bundle owns those
  nodes). The question to MOU should therefore be: *what id will they mint, and
  will they emit a `cpm:forwardConnector` for the dataset — our
  `backwardConnector` must reference exactly that id.* (This also corrects old
  finding E, which called backward/receipt "the receiver of our output"; the
  full correction is in `PROV_TODO.pre-review.md.bak` history.)
- **T13 / T13a** — system boundary ("create dataset" ours or MOU's) and the
  entity-shape decision (now D-d).
- **Meeting decisions D1–D7** (main activity is chain-level; exact id match; two
  dimensions; bind to real data; two training inputs; MOU linkage deferred;
  synthetic data only) — all still in force; full text in the backup file.

## Suggested order

1. **D-a/D-b/D-c calls** (minutes, unblock P0) → then
2. **R10, R11, R12** (document parses + valid) → validate with
   `ProvDocument.deserialize(format='json')`,
3. **R1, R2, R3, R4** (CPM-conformant), re-running self-check after each,
4. **R5, R14/D-d, R15** (Major),
5. P2 hygiene opportunistically (R17/R18/R20/R21 are one-liners each),
6. T7 → T10 → T8 as before; T10b last as the acceptance gate.

## Verification state (carried from T1–T6, pre-review)

- `python provenance.py --self-check` passes, `ruff` clean, `mypy --strict`
  31 pre-existing errors. **But** the self-check asserts several shapes the
  review marks wrong (derivation field names R10, specializationOf direction
  R2, `referencedMetaspectVersion` spelling R4) — fixing P0 without updating
  the self-check will make it fail, which is the point: update assertions and
  code in the same change.
- Real `.png` and real-server chain still unverified (no `dot` binary, no
  network to the tracking server from this sandbox).
- Post-P0 acceptance to add: PROV-JSON deserialization round-trip (see P0).

## History (condensed; details in `PROV_TODO.pre-review.md.bak`)

- **Done:** T1 (env/deps), T2a (id reconciliation, script half), T3
  (placeholders — names now corrected by R4), T4 (diagram/JSON parity), T5
  (backbone relations — specializationOf direction now contested by R2), T6
  (model entity + forward connector).
- **Out of scope by decision:** T2 (single source of truth) — every structural
  change is made twice, guarded by `--self-check`.
- **Standing code findings:** A (export paths diverged — largely addressed by
  T4, keep the self-check), B (assembler never reads per-run `prov.json`s —
  all structural fixes go through `provenance.py`), E (only the sending half of
  the backbone is emittable from mlkit), plus the pre-existing issues list
  (duplicate agent declaration in the diagram path, missing `prov:type`s on
  run activities in the diagram path, README drift, dead CI workflow,
  `positive_label` counting the negative class, callback-ordering pitfall).
