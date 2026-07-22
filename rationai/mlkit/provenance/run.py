"""Training-run PROV-O document generation.

Holds ``build_training_run_prov`` and the constants it needs for mapping
MLflow params/tags to PROV properties.
"""

from __future__ import annotations

import json
from typing import Any

from rationai.mlkit.provenance.common import (
    _iso_timestamp,
    _qualified,
    _qualified_name,
    _safe_id,
    _typed_value,
    get_prov_prefixes,
)


# ──────────────────────────────────────────────
# Hyperparameter keys that surface on the activity
# ──────────────────────────────────────────────

_ACTIVITY_HP_KEYS: set[str] = {
    "learning_rate",
    "lr",
    "batch_size",
    "epochs",
    "optimizer",
    "loss_function",
    "dropout",
    "weight_decay",
    "momentum",
    "num_layers",
    "hidden_size",
    "embedding_dim",
    "num_classes",
    "patch_size",
    "input_size",
    "augmentations",
}

_WSI_PARAM_KEYS: set[str] = {
    "scanner",
    "slide_id",
    "wsi_id",
    "patient_id",
    "subject_id",
    "institution",
    "site",
    "staining",
    "slicing_method",
}


# ──────────────────────────────────────────────
# PROV document builder
# ──────────────────────────────────────────────


def build_training_run_prov(
    run_id: str,
    run_name: str,
    params: dict[str, str],
    metrics: dict[str, float],
    tags: dict[str, str],
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
    split_data: dict[str, object] | None = None,
    requirements: str | None = None,
    verification: dict[str, object] | None = None,
    prov_prefixes: dict[str, str] | None = None,
) -> dict[str, object]:
    """Build an OpenProvenance-compatible PROV document for a training run."""
    username = tags.get("username", tags.get("mlflow.user", "unknown"))
    agent_local = _safe_id(f"user_{username}")
    agent_id = _qualified("gen", agent_local)

    run_act_local = _safe_id(f"run_{run_id}")
    run_act_id = _qualified("gen", run_act_local)

    meta_local = run_id
    meta_id = _qualified("meta", meta_local)

    main_act_local = f"TrainingRun_{run_id[:8]}"
    main_act_id = _qualified("blank", main_act_local)

    entities: dict[str, Any] = {}
    activities: dict[str, Any] = {}
    agents: dict[str, Any] = {}
    used: dict[str, Any] = {}
    was_associated_with: dict[str, Any] = {}

    rel_counter = [0]

    def _blank_rel_id() -> str:
        rid = f"_:n{rel_counter[0]}"
        rel_counter[0] += 1
        return rid

    # ── 1. AGENT ───────────────────────────────────────────
    agent_props: dict[str, Any] = {}
    real_name = tags.get("real_name", username)
    agent_props["schema:name"] = _typed_value(real_name)
    email = tags.get("mlflow.source.git.user.email", f"{username}@unknown")
    agent_props["schema:email"] = _typed_value(email)
    org = tags.get("organization", "")
    if org:
        agent_props["schema:affiliation"] = _typed_value(org)
    agent_props["prov:type"] = [_qualified_name("schema", "Person")]
    agents[agent_id] = agent_props

    # ── 2. INPUT ENTITIES ──────────────────────────────────
    image_path_candidates = (
        params.get("image_path")
        or params.get("wsi_path")
        or params.get("dataset_path")
        or params.get("data_path")
        or params.get("input_path")
    )

    if image_path_candidates:
        wsi_local = _safe_id(f"wsi_{image_path_candidates}")
        wsi_id = _qualified("gen", wsi_local)
        wsi_props: dict[str, Any] = {
            "schema:name": _typed_value(f"Input: {image_path_candidates}"),
            "prov:type": [_qualified_name("sosa", "Sample")],
        }
        if "scanner" in params:
            wsi_props["gen:scanner"] = _typed_value(params["scanner"])
        for pk, prov_key in [
            ("slide_id", "schema:identifier"),
            ("wsi_id", "schema:identifier"),
            ("patient_id", "gen:patient_pseudonym"),
            ("subject_id", "gen:patient_pseudonym"),
            ("institution", "gen:origin_institution"),
            ("site", "gen:origin_institution"),
            ("staining", "gen:staining_method"),
            ("slicing_method", "gen:slicing_method"),
        ]:
            if pk in params:
                wsi_props[prov_key] = _typed_value(params[pk])

        entities[wsi_id] = wsi_props
        used[_blank_rel_id()] = {
            "prov:activity": run_act_id,
            "prov:entity": wsi_id,
        }
    else:
        train_count = params.get("train_samples", "0")
        test_count = params.get("test_samples", "0")
        ds_local = _safe_id(f"dataset_{run_id[:8]}")
        ds_id = _qualified("gen", ds_local)
        entities[ds_id] = {
            "schema:name": _typed_value(
                f"Training dataset ({train_count} train, {test_count} test)"
            ),
            "prov:type": [_qualified_name("sosa", "Sample")],
        }
        used[_blank_rel_id()] = {
            "prov:activity": run_act_id,
            "prov:entity": ds_id,
        }

    # ── 3. RUN ACTIVITY ────────────────────────────────────
    run_activity: dict[str, Any] = {}
    run_activity["prov:type"] = [_qualified_name("schema", "Action")]
    run_activity["prov:startTime"] = [_iso_timestamp(start_time_ms)]
    run_activity["prov:endTime"] = [_iso_timestamp(end_time_ms)]
    run_activity["schema:name"] = _typed_value(run_name)

    exp_name = params.get("model_name", "")
    if exp_name:
        run_activity["gen:experiment_name"] = _typed_value(exp_name)

    if "model_class" in params:
        run_activity["gen:model_config"] = _typed_value(params["model_class"])

    git_commit = tags.get("git_commit", tags.get("mlflow.source.git.commit", ""))
    if git_commit:
        run_activity["schema:identifier"] = _typed_value(git_commit)

    for key in ("pretrained_model", "backbone", "feature_extractor"):
        if key in params:
            run_activity["gen:pretrained_model"] = _typed_value(params[key])

    for key, prov_key in [
        ("dataset_name", "gen:dataset_name"),
        ("dataset_version", "gen:dataset_version"),
        ("data_split", "gen:data_split"),
        ("split", "gen:data_split"),
    ]:
        if key in params:
            run_activity[prov_key] = _typed_value(params[key])

    for key in _ACTIVITY_HP_KEYS:
        if key in params:
            run_activity[f"gen:{key}"] = _typed_value(params[key])

    for key, val in params.items():
        if key.startswith(("opt_", "sch_")):
            clean = key.removeprefix("opt_").removeprefix("sch_")
            if f"gen:{clean}" not in run_activity:
                run_activity[f"gen:{clean}"] = _typed_value(val)

    for tag_key, prov_key in [
        ("mlflow.gpu.count", "gen:gpu_count"),
        ("mlflow.gpu.names", "gen:gpu_names"),
        ("mlflow.cpu.count", "gen:cpu_count"),
        ("mlflow.memory_gb", "gen:memory_gb"),
    ]:
        if tag_key in tags:
            run_activity[prov_key] = _typed_value(tags[tag_key])

    for param_key, prov_key in [
        ("gpu_count", "gen:gpu_count"),
        ("gpu_name", "gen:gpu_names"),
        ("cpu_count_logical", "gen:cpu_count"),
        ("ram_total_gb", "gen:memory_gb"),
    ]:
        if param_key in params and prov_key not in run_activity:
            run_activity[prov_key] = _typed_value(params[param_key])

    git_url = tags.get("git_url", tags.get("mlflow.source.git.remote", ""))
    if git_url:
        run_activity["gen:git_remote"] = _typed_value(git_url)

    source_name = tags.get("mlflow.source.name", "")
    if source_name:
        run_activity["gen:source_name"] = _typed_value(source_name)

    for key, prov_key in [
        ("segmentation", "gen:segmentation_config"),
        ("model", "gen:model_config"),
    ]:
        if key in params:
            run_activity[prov_key] = _typed_value(params[key])

    activities[run_act_id] = run_activity

    # ── 4. CPM METADATA ENTITY ─────────────────────────────
    meta_entity: dict[str, Any] = {}
    meta_entity["prov:type"] = [_qualified_name("cpm", "BundleMetadata")]
    org_val = tags.get("organization", "")
    if org_val:
        meta_entity["cpm:organization"] = _typed_value(org_val)

    skip_keys = (
        set(_ACTIVITY_HP_KEYS)
        | _WSI_PARAM_KEYS
        | {
            "image_path",
            "wsi_path",
            "dataset_path",
            "data_path",
            "input_path",
            "segmentation",
            "model",
            "pretrained_model",
            "backbone",
            "feature_extractor",
            "dataset_name",
            "dataset_version",
            "data_split",
            "split",
            "scanner",
            "slide_id",
            "wsi_id",
            "patient_id",
            "subject_id",
            "institution",
            "site",
            "staining",
            "slicing_method",
        }
    )

    for key, val in params.items():
        if key not in skip_keys:
            safe_key = _safe_id(key)
            meta_entity[f"gen:{safe_key}"] = _typed_value(val)

    for key, mval in metrics.items():
        safe_key = _safe_id(key)
        meta_entity[f"gen:{safe_key}"] = _typed_value(mval)

    if split_data:
        meta_entity["gen:split_test_size"] = _typed_value(
            split_data.get("test_size", "0.2")
        )
        meta_entity["gen:split_random_state"] = _typed_value(
            str(split_data.get("random_state", "42"))
        )
        meta_entity["gen:split_stratified"] = ["true"]

        if split_data.get("train"):
            meta_entity["gen:split_train"] = _typed_value(
                json.dumps(split_data["train"])
            )
        if split_data.get("test"):
            meta_entity["gen:split_test"] = _typed_value(json.dumps(split_data["test"]))

    if requirements:
        meta_entity["gen:requirements"] = [requirements]

    if verification:
        meta_entity["gen:dataset_verified"] = [str(verification.get("verified", False))]
        meta_entity["gen:dataset_run_id"] = [str(verification["dataset_run_id"]) if verification.get("dataset_run_id") is not None else ""]
        fsm = verification.get("file_sizes_match")
        if fsm is not None:
            meta_entity["gen:file_sizes_match"] = [str(fsm)]
        fm = verification.get("files_missing", 0)
        ft = verification.get("files_total", 0)
        meta_entity["gen:files_missing"] = [str(fm)]
        meta_entity["gen:files_total"] = [str(ft)]

    for tag_key in (
        "mlflow.source.git.branch",
        "mlflow.source.git.repo_url",
        "mlflow.parentRunId",
        "mlflow.note.content",
    ):
        if tag_key in tags:
            safe_key = _safe_id(tag_key)
            meta_entity[f"gen:{safe_key}"] = _typed_value(tags[tag_key])

    entities[meta_id] = meta_entity

    was_generated_by: dict[str, Any] = {}
    was_generated_by[_blank_rel_id()] = {
        "prov:entity": meta_id,
        "prov:activity": run_act_id,
    }

    # ── 5. CPM MAIN ACTIVITY ───────────────────────────────
    main_activity: dict[str, Any] = {}
    main_activity["prov:type"] = [_qualified_name("cpm", "mainActivity")]
    main_activity["cpm:referencedMetaBundleId"] = [
        {"type": "prov:QUALIFIED_NAME", "$": meta_id},
    ]
    main_activity["dct:hasPart"] = [
        {"type": "prov:QUALIFIED_NAME", "$": run_act_id},
    ]
    activities[main_act_id] = main_activity

    # ── 6. RELATIONSHIPS ───────────────────────────────────
    was_associated_with[_blank_rel_id()] = {
        "prov:activity": run_act_id,
        "prov:agent": agent_id,
    }

    # ── 7. ASSEMBLE BUNDLE ─────────────────────────────────
    inner: dict[str, object] = {"prefix": prov_prefixes or get_prov_prefixes()}
    if entities:
        inner["entity"] = entities
    if activities:
        inner["activity"] = activities
    if agents:
        inner["agent"] = agents
    if was_associated_with:
        inner["wasAssociatedWith"] = was_associated_with
    if was_generated_by:
        inner["wasGeneratedBy"] = was_generated_by
    if used:
        inner["used"] = used

    bundle_key = f"storage:{run_id}"
    return {"bundle": {bundle_key: inner}}
