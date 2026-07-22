"""User registration with PROV-O document generation.

Colocates ``build_user_prov`` (the PROV document builder) and
``register_new_user`` (the MLflow registration entry point).
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from typing import Any

import mlflow

from rationai.mlkit.provenance.common import (
    _iso_timestamp,
    _qualified,
    _qualified_name,
    _safe_id,
    _typed_value,
    get_prov_prefixes,
)


# ──────────────────────────────────────────────
# PROV document builder
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
        {"type": "prov:QUALIFIED_NAME", "$": meta_id},
    ]
    main_activity["dct:hasPart"] = [
        {"type": "prov:QUALIFIED_NAME", "$": run_act_id},
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


# ──────────────────────────────────────────────
# MLflow registration entry point
# ──────────────────────────────────────────────


def register_new_user(
    username: str,
    real_name: str,
    email: str,
    organization: str,
    lead_name: str,
    lead_email: str,
) -> str:
    """Register a user and emit a PROV-O document as an artifact.

    Creates a run in the ``User_Registry`` experiment with identity tags
    and a W3C PROV-O ``prov.json`` document compatible with the Java
    ``prov_mlflow`` tool.

    Returns:
        The MLflow run_id of the registration run.
    """
    experiment_name = "User_Registry"
    experiment_id: str | None
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except Exception:
        exp = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = exp.experiment_id if exp else None

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"User_{username}",
    ) as run:
        run_id = run.info.run_id

        mlflow.log_params(
            {
                "username": username,
                "real_name": real_name,
                "email": email,
                "organization": organization,
                "lead_name": lead_name,
                "lead_email": lead_email,
            }
        )
        mlflow.set_tags(
            {
                "username": username,
                "organization": organization,
            }
        )

        # ── PROV-O document ────────────────────────────────
        prov_doc = build_user_prov(
            run_id=run_id,
            username=username,
            real_name=real_name,
            email=email,
            organization=organization,
            lead_name=lead_name,
            lead_email=lead_email,
        )

        prov_dir = f"_user_prov_{uuid.uuid4().hex[:8]}"
        os.makedirs(prov_dir, exist_ok=True)
        try:
            prov_path = os.path.join(prov_dir, "prov.json")
            with open(prov_path, "w") as f:
                json.dump(prov_doc, f, indent=2)

            mlflow.log_artifact(prov_path, artifact_path="provenance")
        finally:
            shutil.rmtree(prov_dir, ignore_errors=True)

    print(f"  [register_new_user] {username} → run_id={run_id}")
    return run_id


if __name__ == "__main__":
    register_new_user(
        username="researcher_01",
        real_name="Jane Doe",
        email="jane.doe@example.com",
        organization="Example Org",
        lead_name="John Smith",
        lead_email="john.smith@example.com",
    )
