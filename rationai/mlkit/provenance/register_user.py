"""Register a researcher into MLflow's User_Registry experiment."""

from __future__ import annotations

import json
import os
import shutil
import uuid

import mlflow


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
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if not experiment:
        mlflow.create_experiment(experiment_name)

    with mlflow.start_run(
        experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id,
        run_name=f"User_{username}",
    ) as run:
        run_id = run.info.run_id

        mlflow.set_tags({
            "username": username,
            "real_name": real_name,
            "email": email,
            "organization": organization,
            "lead_name": lead_name,
            "lead_email": lead_email,
        })

        # ── PROV-O document ────────────────────────────────
        from rationai.mlkit.provenance.prov import build_user_prov  # noqa: PLC0415

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
        prov_path = os.path.join(prov_dir, "prov.json")
        with open(prov_path, "w") as f:
            json.dump(prov_doc, f, indent=2)

        mlflow.log_artifact(prov_path, artifact_path="provenance")
        shutil.rmtree(prov_dir, ignore_errors=True)

    print(f"  [register_new_user] {real_name} ({username}) → run_id={run_id}")
    return run_id


if __name__ == "__main__":
    register_new_user(
        username="jiribuchta",
        real_name="Jiří Buchta",
        email="524981@mail.muni.cz",
        organization="RationAI",
        lead_name="Tomáš Brázdil",
        lead_email="brazdil@muni.cz",
    )
