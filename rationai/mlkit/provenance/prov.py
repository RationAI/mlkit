"""Shared PROV-O document builder.

Produces OpenProvenance-compatible JSON bundles compatible with the
Java ``prov_mlflow`` tool and used by both registration runs and
training provenance callbacks.
"""

from __future__ import annotations

import json
import os
import re
import datetime as _dt
from typing import Any


# ──────────────────────────────────────────────
# OpenProvenance / CPM namespace URIs
# ──────────────────────────────────────────────

_DEFAULT_PROV_PREFIXES: dict[str, str] = {
    "storage": "http://localhost:8083/api/v1/documents/",
    "meta": "http://localhost:8083/api/v1/documents/meta/",
    "schema": "https://schema.org/",
    "cpm": "https://www.commonprovenancemodel.org/cpm-namespace-v1-0/",
    "blank": "https://openprovenance.org/blank/",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "gen": "gen/",
    "dct": "http://purl.org/dc/terms/",
    "prov": "http://www.w3.org/ns/prov#",
    "sosa": "http://www.w3.org/ns/sosa/",
}


def get_prov_prefixes(override: dict[str, str] | None = None) -> dict[str, str]:
    """Return PROV prefix map.

    Priority: explicit override > ``PROV_BASE_URI`` env var > defaults.
    """
    if override:
        return {**_DEFAULT_PROV_PREFIXES, **override}
    env_json = os.environ.get("PROV_BASE_URI", "")
    if env_json:
        try:
            merged = {**_DEFAULT_PROV_PREFIXES, **json.loads(env_json)}
            return merged
        except json.JSONDecodeError:
            pass
    return _DEFAULT_PROV_PREFIXES


# ──────────────────────────────────────────────
# Small helpers used inside the PROV document
# ──────────────────────────────────────────────

def _safe_id(name: str) -> str:
    """Sanitise a name so it can be used as a PROV identifier fragment."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


def _qualified(prefix: str, local: str) -> str:
    return f"{prefix}:{local}"


def _typed_value(value: Any) -> list[str]:
    return [str(value)]


def _qualified_name(type_prefix: str, type_local: str) -> dict[str, str]:
    return {"type": "prov:QUALIFIED_NAME", "$": f"{type_prefix}:{type_local}"}


def _iso_timestamp(ts_ms: int | None = None) -> str:
    if ts_ms is not None:
        dt = _dt.datetime.fromtimestamp(ts_ms / 1000, tz=_dt.UTC)
    else:
        dt = _dt.datetime.now(_dt.UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000+00:00")


# ──────────────────────────────────────────────
# PROV document builders
# ──────────────────────────────────────────────

def build_user_prov(
    run_id: str,
    username: str,
    real_name: str,
    email: str,
    organization: str,
    lead_name: str | None = None,
    lead_email: str | None = None,
    prov_prefixes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a PROV document for a user registration run.

    Produces an ``agent`` entity representing the researcher and links it
    to a *registration activity* that generated the run's metadata bundle.
    """
    prefixes = prov_prefixes or get_prov_prefixes()

    agent_local = _safe_id(f"user_{username}")
    agent_id = _qualified("gen", agent_local)

    run_act_local = _safe_id(f"run_{run_id}")
    run_act_id = _qualified("gen", run_act_local)

    meta_local = run_id
    meta_id = _qualified("meta", meta_local)

    main_act_local = f"UserReg_{run_id[:8]}"
    main_act_id = _qualified("blank", main_act_local)

    entities: dict[str, dict[str, Any]] = {}
    activities: dict[str, dict[str, Any]] = {}
    agents: dict[str, dict[str, list[Any]]] = {}
    was_associated_with: dict[str, dict[str, str]] = {}
    was_generated_by: dict[str, dict[str, str]] = {}

    rel_counter = [0]
    def _blank_rel_id() -> str:
        rid = f"_:n{rel_counter[0]}"
        rel_counter[0] += 1
        return rid

    now = _iso_timestamp()

    # ── AGENT ──────────────────────────────────────────────
    agent_props: dict[str, list[Any]] = {}
    agent_props["schema:name"] = _typed_value(real_name)
    agent_props["schema:email"] = _typed_value(email)
    if organization:
        agent_props["schema:affiliation"] = _typed_value(organization)
    agent_props["prov:type"] = [_qualified_name("schema", "Person")]
    agents[agent_id] = agent_props

    # ── ACTIVITY (the registration action) ────────────────
    run_activity: dict[str, Any] = {}
    run_activity["prov:type"] = [_qualified_name("schema", "Action")]
    run_activity["prov:startTime"] = [now]
    run_activity["prov:endTime"] = [now]
    run_activity["schema:name"] = _typed_value(f"Register user {real_name}")
    activities[run_act_id] = run_activity

    # ── CPM METADATA ENTITY ───────────────────────────────
    meta_entity: dict[str, list[Any]] = {}
    meta_entity["prov:type"] = [_qualified_name("cpm", "BundleMetadata")]
    meta_entity["gen:username"] = _typed_value(username)
    meta_entity["gen:real_name"] = _typed_value(real_name)
    meta_entity["gen:email"] = _typed_value(email)
    if organization:
        meta_entity["gen:organization"] = _typed_value(organization)
    if lead_name:
        meta_entity["gen:lead_name"] = _typed_value(lead_name)
    if lead_email:
        meta_entity["gen:lead_email"] = _typed_value(lead_email)
    entities[meta_id] = meta_entity

    # ── CPM MAIN ACTIVITY ────────────────────────────────
    main_activity: dict[str, Any] = {}
    main_activity["prov:type"] = [_qualified_name("cpm", "mainActivity")]
    main_activity["cpm:referencedMetaBundleId"] = [
        {"type": "prov:QUALIFIED_NAME", "$": meta_id}
    ]
    main_activity["dct:hasPart"] = [
        {"type": "prov:QUALIFIED_NAME", "$": run_act_id}
    ]
    activities[main_act_id] = main_activity

    # ── RELATIONSHIPS ─────────────────────────────────────
    was_associated_with[_blank_rel_id()] = {
        "prov:activity": run_act_id,
        "prov:agent": agent_id,
    }
    was_generated_by[_blank_rel_id()] = {
        "prov:entity": meta_id,
        "prov:activity": run_act_id,
    }

    # ── ASSEMBLE BUNDLE ───────────────────────────────────
    inner: dict[str, Any] = {"prefix": prefixes}
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

    bundle_key = f"storage:{run_id}"
    return {"bundle": {bundle_key: inner}}


def build_dataset_prov(
    run_id: str,
    dataset_name: str,
    version: str,
    dataset_root: str,
    num_samples: int,
    num_positive: int,
    num_negative: int,
    file_sizes: dict[str, int],
    manifest_path: str | None = None,
    prov_prefixes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a PROV document for a dataset registration run.

    Produces a ``dataset`` entity (``sosa:Sample``) linked to the
    registration activity and metadata bundle.
    """
    prefixes = prov_prefixes or get_prov_prefixes()

    run_act_local = _safe_id(f"run_{run_id}")
    run_act_id = _qualified("gen", run_act_local)

    ds_local = _safe_id(f"dataset_{dataset_name}_{version.replace('.', '_')}")
    ds_id = _qualified("gen", ds_local)

    meta_local = run_id
    meta_id = _qualified("meta", meta_local)

    main_act_local = f"DatasetReg_{run_id[:8]}"
    main_act_id = _qualified("blank", main_act_local)

    entities: dict[str, dict[str, Any]] = {}
    activities: dict[str, dict[str, Any]] = {}
    used: dict[str, dict[str, Any]] = {}
    was_generated_by: dict[str, dict[str, Any]] = {}

    rel_counter = [0]
    def _blank_rel_id() -> str:
        rid = f"_:n{rel_counter[0]}"
        rel_counter[0] += 1
        return rid

    now = _iso_timestamp()

    # ── DATASET ENTITY ───────────────────────────────────
    ds_props: dict[str, list[Any]] = {
        "schema:name": _typed_value(dataset_name),
        "prov:type": [_qualified_name("sosa", "Sample")],
        "dct:description": _typed_value(
            f"Dataset {dataset_name} v{version} ({num_samples} samples)"
        ),
    }
    if manifest_path:
        ds_props["schema:url"] = _typed_value(manifest_path)
    entities[ds_id] = ds_props

    # ── ACTIVITY (the registration action) ────────────────
    run_activity: dict[str, Any] = {}
    run_activity["prov:type"] = [_qualified_name("schema", "Action")]
    run_activity["prov:startTime"] = [now]
    run_activity["prov:endTime"] = [now]
    run_activity["schema:name"] = _typed_value(f"Register dataset {dataset_name}")
    run_activity["gen:dataset_name"] = _typed_value(dataset_name)
    run_activity["gen:dataset_version"] = _typed_value(version)
    activities[run_act_id] = run_activity

    # ── USED (activity consumed the dataset entity) ──────
    used[_blank_rel_id()] = {
        "prov:activity": run_act_id,
        "prov:entity": ds_id,
    }

    # ── CPM METADATA ENTITY ───────────────────────────────
    meta_entity: dict[str, list[Any]] = {}
    meta_entity["prov:type"] = [_qualified_name("cpm", "BundleMetadata")]
    meta_entity["gen:dataset_name"] = _typed_value(dataset_name)
    meta_entity["gen:dataset_version"] = _typed_value(version)
    meta_entity["gen:dataset_root"] = _typed_value(dataset_root)
    meta_entity["gen:num_samples"] = _typed_value(str(num_samples))
    meta_entity["gen:num_positive"] = _typed_value(str(num_positive))
    meta_entity["gen:num_negative"] = _typed_value(str(num_negative))
    if manifest_path:
        meta_entity["gen:manifest_path"] = _typed_value(manifest_path)

    # Store per-file sizes as a single string value (matches prov_mlflow convention)
    file_sizes_str = json.dumps(file_sizes)
    meta_entity["gen:file_sizes"] = [file_sizes_str]

    entities[meta_id] = meta_entity

    # ── CPM MAIN ACTIVITY ────────────────────────────────
    main_activity: dict[str, Any] = {}
    main_activity["prov:type"] = [_qualified_name("cpm", "mainActivity")]
    main_activity["cpm:referencedMetaBundleId"] = [
        {"type": "prov:QUALIFIED_NAME", "$": meta_id}
    ]
    main_activity["dct:hasPart"] = [
        {"type": "prov:QUALIFIED_NAME", "$": run_act_id}
    ]
    activities[main_act_id] = main_activity

    # ── RELATIONSHIPS ─────────────────────────────────────
    was_generated_by[_blank_rel_id()] = {
        "prov:entity": meta_id,
        "prov:activity": run_act_id,
    }

    # ── ASSEMBLE BUNDLE ───────────────────────────────────
    inner: dict[str, Any] = {"prefix": prefixes}
    if entities:
        inner["entity"] = entities
    if activities:
        inner["activity"] = activities
    if used:
        inner["used"] = used
    if was_generated_by:
        inner["wasGeneratedBy"] = was_generated_by

    bundle_key = f"storage:{run_id}"
    return {"bundle": {bundle_key: inner}}
