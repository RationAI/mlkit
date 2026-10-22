# PROV Revision Plan — 2026-10-20

> **What this is.** Working notes for the next round of changes to the mlkit
> provenance output, driven by Rudolf Wittner (CPM/PROV expert). Two sources:
> (a) Rudolf's **13 revision notes** from the chat (8:51–9:41), and (b) the
> **meeting transcript** where Šimon + Rudolf walked through them.
>
> **Scope of this round (per the meeting wrap-up):** implement the 13 notes →
> the `.prov.json` is "done." **Storage deployment** (meeting with Jan, Wed) and
> **slide/model hashing** (call with Jirka, Mon) are *next*, not this round.
>
> Status legend: ✅ confirmed, implement now · ⏳ needs one more confirmation.

---

## The 13 notes

### 1. "uppercase in prov types" ✅
- **Meaning:** CPM type names must be **PascalCase** — that's how they're defined in
  the namespace. The camelCase CPM terms are the only non-conforming `prov:type` values.
- **Implement:**
  - `cpm:mainActivity` → `cpm:MainActivity` — [provenance.py:169](provenance.py#L169)
  - `CPM_CURRENT_CONNECTOR` → `cpm:CurrentConnector`, `CPM_FORWARD_CONNECTOR` →
    `cpm:ForwardConnector`, `CPM_CURRENT_AGENT` → `cpm:CurrentAgent` —
    [provenance.py:122-126](provenance.py#L122)
  - Update the matching self-check assertions.
  - Leave standard types as-is (`prov:Person`, `prov:Entity`, `sosa:Sample`,
    `schema:Action` … are already correct).
- **Related rule (from meeting):** only emit CPM terms that are *actually defined*
  in the namespace.

### 2. "správný namespace uri → `https://www.commonprovenancemodel.org/ns/cpm#`" ✅
- **Meaning:** **revises R1.** The correct namespace has `/ns/` in it.
- **Implement:**
  - `PREFIX["cpm"]` + `NAMESPACE_URIS["cpm"]`: `…/cpm#` → `…/ns/cpm#` —
    [provenance.py:60,77](provenance.py#L60)
  - `cpm` entry in the [common.py](rationai/mlkit/provenance/common.py) prefix map
    (same change, so both builders agree).

### 3. "ISO-IS-23494-2:2026 instead of metabundleSpecV" ✅
- **Meaning:** real value for `cpm:referencedMetaBundleSpecV` = the ISO/IEC 23494 CPM
  standard. Closes D-e.
- **Implement:** `CPM_PLACEHOLDERS["cpm:referencedMetaBundleSpecV"]` =
  `"ISO-IS-23494-2:2026"` — [provenance.py:104](provenance.py#L104).
  (`referencedMetaBundleId` stays a placeholder → the R5 store URL.)
- **Minor confirm:** use Rudolf's exact string `ISO-IS-23494-2:2026`, or normalize to
  `ISO/IEC 23494-2:2026`?

### 4. "folder = collection" ✅
- **Meaning:** run-output *folders* (`gen:output_*`, the `…/artifacts` ones) are
  `prov:Collection`; the files inside are their members.
- **Implement:**
  - `output_props` non-model branch: `sosa:Sample` → `prov:Collection` —
    [provenance.py:287](provenance.py#L287) (duplicates at
    [provenance.py:619](provenance.py#L619), [provenance.py:918](provenance.py#L918)).
  - Add `hadMember(folder, file)` edges — this is where `hadMember` comes in (note 5).
- **Confirm:** *replace* `sosa:Sample`, or keep both types?

### 5. "add derived from (output derived or not from input)" ✅ (clarified by meeting)
- **Meaning — the key clarification from the transcript.** The current
  `wasDerivedFrom(input_file, upstream_output_folder)` edge is a **misuse**: the file
  is not *derived from* the folder, it's *in* it. Split that one edge into two
  correct relations:
  1. **`hadMember(folder, file)`** — the collection contains the file (Rudolf:
     *"in that place, hadMember fits"*).
  2. **`wasDerivedFrom(output, input)`** — the real cross-run lineage: each run's
     output is derived from the input it consumed.
- **Implement:** the current misused edge is
  [provenance.py:634-636](provenance.py#L634) (JSON) and
  [provenance.py:927](provenance.py#L927) (diagram). Replace with
  `hadMember(out_of(up), inp)` **and** add `wasDerivedFrom(out_of(run), inp)` per run.
- **Confirm:** in PROV, `used` + `wasGeneratedBy` already *justify* a derivation; the
  explicit `wasDerivedFrom(output, input)` is an abridgement on top. Default plan:
  **add** the explicit edges (since "add derived from" is an instruction).

### 6. "mlkit prefix # params resolvable" ✅ (likely a no-op)
- **Meaning:** `mlkit:` should end in `#` so `mlkit:params` is a resolvable IRI
  (`…/mlkit#params`).
- **Implement:** already the case — `"mlkit": "http://rationai.org/prov/mlkit#"`
  ([provenance.py:67](provenance.py#L67)). Confirm it's a no-op, or whether a
  `#`-less variant was seen somewhere.

### 7. "if activity can run without existing entities then they are not used" ⏳
- **Meaning:** fixes R15 over-approximation — only emit `used` for entities a run
  actually needs; drop config-URIs the run can execute without.
- **Implement:** the input/`used` loop — [provenance.py:609-623](provenance.py#L609).
- **OPEN:** the *rule* is clear, but the transcript **flags the open question**
  (Šimon: *"how do we recognize when…"*). Need a decision on **how** to tell which
  config-URIs a run really reads — a producer list, or a coded heuristic.

### 8. "add dataset entity, no backward connector yet" ✅
- **Meaning:** R15 under-approximation. Add the slide/dataset entity (a Collection),
  `used` by **Dataset Creation**. **No backwardConnector yet** (T14 still pending MOU).
- **Implement:** new entity + one `used` edge on the input side of the chain.
- **Confirm:** type it `prov:Collection` (consistent with note 4)? What URI/name for
  the slide set?

### 9. "add any uri to gen prefix" ✅
- **Meaning:** R18. Give `gen:` a real absolute IRI ("any uri" = our pick).
- **Implement:** `PREFIX["gen"]` + `NAMESPACE_URIS["gen"]`: `"gen/"` →
  e.g. `http://rationai.org/prov/gen#` (matches the mlkit base) —
  [provenance.py:65,73](provenance.py#L65) + common.py.

### 10. "storage instead of blank prefix (do traversal information)" ✅
- **Meaning:** R20. The main activity (the traversal anchor) uses `storage:` instead
  of `blank:`, so it's addressable in the CPM store.
- **Implement:** `main_activity_id`: `blank:Run_…` → `storage:…` —
  [provenance.py:183](provenance.py#L183).
- **Gotcha:** make sure `storage` is in `NAMESPACE_URIS` (it's in `PREFIX`), or the
  prov-diagram path's qualified-name expansion won't resolve `storage:` ids.

### 11. "vše co je v traversal info add storage prefix" ⏳
- **Meaning:** everything in the "traversal info" (the CPM backbone a reader walks)
  gets the `storage:` prefix.
- **Implement:** `forward_connector_id` + `current_connector_id`: `gen:…` →
  `storage:…` — [provenance.py:193,205](provenance.py#L193).
- **OPEN:** confirm the exact node set. Working read: **mainActivity +
  forwardConnector + currentConnector**. Plain domain `gen:` entities
  (`gen:input_/output_/model_`) and agents (`gen:user_`) stay in `gen:`.

### 12. "add external id" ✅ (clarified by meeting)
- **Meaning:** R21. `cpm:externalId` = the **MLflow URL** (artifact/run URI) — the
  value by which you can locate the object in MLflow or on the filesystem (*"the best
  and only option"*). `cpm:externalIdType` ≈ `"MLflow artifact URI"`.
- **Implement:** **additive** — attach `cpm:externalId` + `cpm:externalIdType` on the
  run activities and the artefact entities, both builder paths. Keep the `gen:` names.

### 13. "remove meta node" ✅ (confirmed by meeting)
- **Meaning:** R22. Delete the `cpm:BundleMetadata` `meta:` entity **entirely**.
  Rudolf's reasons: `cpm:BundleMetadata` is an *undefined* term;
  `cpm:organization`-as-agent is undefined; the `gen:input_uris` **duplicate**
  information; no added value.
- **Implement:** remove the meta entity block —
  [provenance.py:643-652](provenance.py#L643) (JSON) and
  [provenance.py:929-943](provenance.py#L929) (diagram). The main activity keeps its
  two mandatory attributes; `referencedMetaBundleId` then falls back to its
  placeholder → the R5 store URL; `referencedMetaBundleSpecV` = ISO value (note 3).
  Also drops the self-check assertion at
  [provenance.py:1354-1355](provenance.py#L1354).

---

## What the meeting transcript settled
- **`hadMember` vs `wasDerivedFrom`** (notes 4+5): the current
  `wasDerivedFrom(file, folder)` is a misuse → `hadMember` for containment + a real
  `wasDerivedFrom(output, input)` for lineage.
- **external id = the MLflow URL** (note 12).
- **Remove the meta node entirely** (note 13), plus the general rule: *only use CPM
  terms that are actually defined.*
- **Hashing is out of scope** this round (Monday call with Jirka).
- **Scope:** 13 notes → JSON "done" → then storage (Wednesday, with Jan).

---

## Open questions (resolve before final implementation)
1. **Note 7** — how to determine "can run without"? (producer list vs. heuristic)
2. **Note 11** — confirm the traversal-info node set (mainActivity + 2 connectors?).
3. **Note 5** — add explicit `wasDerivedFrom(output, input)` abridgement edges, or is
   `hadMember` + `used`/`generated` enough? *(default: add them)*
4. **Note 4** — `prov:Collection` *replace* or *in addition to* `sosa:Sample`?
5. **Note 3** — exact ISO string format (`ISO-IS-23494-2:2026` vs `ISO/IEC …`)?
6. **Note 8** — dataset entity type + its URI/name?
7. **Note 6** — confirm the `mlkit#` `#` is already correct (no-op)?

---

## Suggested implementation order
- **Batch 1 (unambiguous — implement now):** 1, 2, 3, 4(+`hadMember`), 5, 8, 9, 10,
  12, 13. Both builder paths + self-check, then regenerate and verify text-only.
- **Batch 2 (after confirmations):** 7, 11 (+ whichever sub-confirms above are still
  open).

## Verification (unchanged from last round)
- `python provenance.py --self-check` (offline, stub-driven).
- Regenerate real output from the in-cluster MLflow URL
  (`http://mlflow-jiribuchta.rationai-mlflow:5000`), PNG needs
  `export PATH="$HOME/local/bin:$PATH"`. Confirm the PNG is real graphviz (`file …`).
