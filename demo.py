#!/usr/bin/env python3
"""
Demo script for rationai.mlkit — full end-to-end pipeline.

Creates dummy data, registers it in Dataset_Registry, trains a model with
provenance tracking (fail_fast=True), logs everything to MLflow, and prints
a summary of what was uploaded.

Run:
    python demo.py                  # full pipeline (local file store)
    python demo.py --uri http://... # custom MLflow server
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import logging
logging.getLogger("root").setLevel(logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.ERROR)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def sep(title=""):
    print(f"\n{'='*60}")
    if title:
        print(f"  {title}")
        print(f"{'='*60}")


def sub(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ──────────────────────────────────────────────
# 1. Create dummy datasets
# ──────────────────────────────────────────────

def step_create_datasets(n=2, wsis_per_ds=10):
    sub("1. Creating dummy pathology datasets")

    from dummy_dataset_create import create_dummy_datasets
    create_dummy_datasets(
        num_datasets=n,
        wsis_per_dataset=wsis_per_ds,
        data_dir=Path("test_data"),
        seed=42,
        img_size=64,
        clean=True,
    )

    data_dir = Path("test_data")
    for d in sorted(data_dir.iterdir()):
        if d.is_dir():
            manifest = d / "manifest.csv"
            n_rows = sum(1 for _ in open(manifest)) - 1 if manifest.exists() else "?"
            print(f"  → {d.name}/  ({n_rows} samples)")


# ──────────────────────────────────────────────
# 2. Register dataset(s) in Dataset_Registry
# ──────────────────────────────────────────────

def step_register_datasets():
    sub("2. Registering datasets in Dataset_Registry")

    from rationai.mlkit.provenance import register_dataset

    data_dir = Path("test_data")
    for d in sorted(data_dir.iterdir()):
        if d.is_dir():
            manifest = d / "manifest.csv"
            if not manifest.exists():
                continue
            ds_name = d.name
            run_id = register_dataset(
                dataset_dir=str(d),
                dataset_name=ds_name,
            )
            print(f"  ✅ {ds_name}: registered (run={run_id[:8]})")


# ──────────────────────────────────────────────
# 3. Stream capture demo
# ──────────────────────────────────────────────

def step_stream_capture():
    sub("3. StreamCapture + StreamModifier")

    import io
    from rationai.mlkit import StreamCapture, StreamLogger, StreamModifier

    # StreamCapture — captures stdout into a logger
    class _Buf(StreamLogger):
        def __init__(self):
            self._buf = io.StringIO()
        def log_stream(self, text: str):
            self._buf.write(text)
        def get_value(self):
            return self._buf.getvalue()

    logger = _Buf()
    with StreamCapture(logger, streams=(sys.stdout,)):
        print("Hello from stdout!")
        print("\033[92mThis is green (ANSI)\033[0m")

    captured = logger.get_value()
    has_ansi = "\x1b" in captured
    print(f"  Captured text : {captured.strip()!r}")
    print(f"  ANSI preserved: {'✅ (raw capture)' if has_ansi else '❌'}")

    # StreamModifier — injects side-effect logic into a stream's write
    buf = io.StringIO()
    side_log = []
    modifier = StreamModifier(stream=buf, id=42)
    modifier.set_write(lambda s, iid: side_log.append(f"[{iid}] {s}"))
    buf.write("hello")
    modifier.teardown()
    print(f"  Side log      : {side_log}")
    print(f"  Original buf  : {buf.getvalue()!r}")
    print("  ✅ StreamModifier works (side-effect injected before original write)")


# ──────────────────────────────────────────────
# 4. Metrics demo (AggregatedMetricCollection)
#    — skipped if rationai.masks is not installed
# ──────────────────────────────────────────────

def step_metrics():
    sub("4. AggregatedMetricCollection — tile → slide aggregation")

    try:
        import torch
        from torchmetrics import Accuracy
        from rationai.mlkit import AggregatedMetricCollection, MaxAggregator, MeanAggregator
    except ModuleNotFoundError as e:
        if "rationai.masks" in str(e):
            print("  ⊘ SKIPPED — rationai.masks (private dep) not installed")
            return
        raise

    preds = torch.tensor([0.1, 0.8, 0.3, 0.9])
    targets = torch.tensor([0, 1, 0, 1])
    keys = ["slide_A", "slide_A", "slide_B", "slide_B"]

    for name, agg in [("MaxAggregator", MaxAggregator()), ("MeanAggregator", MeanAggregator())]:
        mc = AggregatedMetricCollection(
            metrics={"accuracy": Accuracy(task="binary")},
            aggregator=agg,
        )
        mc.update(preds, targets, keys)
        result = mc.compute()
        print(f"  {name:20s}: accuracy = {result['accuracy'].item():.4f}")


# ──────────────────────────────────────────────
# 5. NestedMetricCollection demo
#    — skipped if rationai.masks is not installed
# ──────────────────────────────────────────────

def step_nested_metrics():
    sub("5. NestedMetricCollection — per-slide multiclass metrics")

    try:
        import torch
        from torchmetrics import Accuracy, Precision
        from rationai.mlkit import NestedMetricCollection
    except ModuleNotFoundError as e:
        if "rationai.masks" in str(e):
            print("  ⊘ SKIPPED — rationai.masks (private dep) not installed")
            return
        raise

    metrics = NestedMetricCollection(
        metrics={
            "accuracy": Accuracy(task="multiclass", num_classes=3),
            "precision": Precision(task="multiclass", num_classes=3, average=None),
        },
        key_name="slide",
        class_names=["benign", "low_grade", "high_grade"],
    )

    preds = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [0.2, 0.6, 0.2]])
    targets = torch.tensor([0, 2, 1])
    keys = ["slide_1", "slide_1", "slide_2"]

    metrics.update(preds, targets, keys)
    result = metrics.compute()

    print(f"  Slides: {result['slide']}")
    for k, v in result.items():
        if k != "slide":
            print(f"  {k:15s}: {v}")


# ──────────────────────────────────────────────
# 6. StratifiedBatchSampler demo
# ──────────────────────────────────────────────

def step_sampler():
    sub("6. StratifiedBatchSampler — balanced class batches")

    from rationai.mlkit import StratifiedBatchSampler, PDMStratifiedBatchSampler
    import pandas as pd

    # List-of-lists sampler
    sampler = StratifiedBatchSampler(
        data_indices=[[0, 1, 2, 3], [4, 5, 6, 7]],
        batch_size=4,
    )
    for i, batch in enumerate(sampler):
        print(f"  Batch {i}: {batch}")
    print(f"  Total batches: {len(sampler)}")

    # DataFrame-based sampler
    df = pd.DataFrame({
        "idx": list(range(8)),
        "label": [0, 0, 0, 1, 1, 1, 1, 0],
    })
    pdm_sampler = PDMStratifiedBatchSampler(data=df, stratify_by="label", batch_size=4)
    pdm_batches = list(pdm_sampler)
    print(f"  PDM sampler batches: {len(pdm_batches)}")


# ──────────────────────────────────────────────
# 7. Full training run with provenance autolog
#    (fail_fast=True — dataset verification must pass)
# ──────────────────────────────────────────────

def step_provenance():
    sub("7. Provenance — full training run, logged to MLflow (fail_fast=True)")

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from rationai.mlkit.provenance import autolog

    class _DummyDS(Dataset):
        def __len__(self):
            return 32
        def __getitem__(self, idx):
            return torch.randn(64), torch.randint(0, 2, (1,)).item()

    @autolog(model_name="demo_model", experiment_name="Demo_Experiment", fail_fast=True)
    def train(run):
        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        run.register_model(model)

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        run.register_optimizer(optimizer)

        loader = DataLoader(_DummyDS(), batch_size=8)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, 5):
            model.train()
            total_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg = total_loss / len(loader)
            run.log_metrics({"train_loss": avg}, step=epoch)
            print(f"    Epoch {epoch}: loss={avg:.4f}")

        run.save_model(model)

    train()

    import mlflow as _mlf
    while _mlf.active_run():
        _mlf.end_run()


# ──────────────────────────────────────────────
# 8. Lightning integration demo
# ──────────────────────────────────────────────

def step_lightning():
    sub("8. Lightning — Trainer + MLFlowLogger (actual training)")

    import torch
    import lightning as pl
    import mlflow
    from rationai.mlkit import Trainer, MLFlowLogger, MultiloaderLifecycle

    class _TinyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Linear(8, 2)
        def forward(self, x):
            return self.net(x)
        def training_step(self, batch, _):
            x = torch.randn(4, 8, device=self.device)
            loss = self.net(x).sum()
            self.log("train_loss", loss)
            return loss
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.01)

    mlflow.set_experiment("Demo_Lightning")
    run = mlflow.start_run(run_name="Training_demo_lightning_model")

    try:
        model = _TinyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        logger = MLFlowLogger(run_id=run.info.run_id)
        trainer = Trainer(
            logger=logger,
            max_epochs=2,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            log_every_n_steps=1,
        )

        print("  Training a tiny Lightning model...")
        trainer.fit(model, torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.randn(16, 8), torch.randint(0, 2, (16,)))
        ))
        print("  ✅ Training complete — check MLflow for logs")
    finally:
        mlflow.end_run()


# ──────────────────────────────────────────────
# 9. Summary — list all runs & artifacts in MLflow
# ──────────────────────────────────────────────

def step_summary():
    sub("9. MLflow Summary")

    import mlflow

    tracking_uri = mlflow.get_tracking_uri()
    print(f"  Tracking URI: {tracking_uri}")

    client = mlflow.MlflowClient()

    for exp_name in ["Dataset_Registry", "Demo_Experiment", "Demo_Lightning"]:
        exp = client.get_experiment_by_name(exp_name)
        if not exp:
            continue
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["attributes.start_time desc"],
        )
        print(f"\n  Experiment: {exp_name}  ({len(runs)} run(s))")

        for r in runs[:5]:
            params = dict(r.data.params) if r.data.params else {}
            metrics = dict(r.data.metrics) if r.data.metrics else {}
            artifacts = [a.path for a in client.list_artifacts(r.info.run_id)]

            print(f"\n    Run: {r.info.run_name or r.info.run_id[:8]}")
            print(f"      Status : {r.info.status}")
            if params:
                print(f"      Params : {params}")
            if metrics:
                print(f"      Metrics: {metrics}")

            # Group artifacts by folder
            folders = {}
            for a in artifacts:
                folder = a.split("/")[0] if "/" in a else ""
                folders.setdefault(folder, []).append(a)
            for folder, files in sorted(folders.items()):
                print(f"      [{folder}] {', '.join(os.path.basename(f) for f in files)}")

            # Check provenance summary
            prov_path = None
            for a in artifacts:
                if "run_summary.json" in a:
                    prov_path = a
                    break
            if prov_path:
                local = client.download_artifacts(r.info.run_id, prov_path)
                with open(local) as f:
                    summary = json.load(f)
                print(f"      Provenance keys: {', '.join(summary.keys())}")

                if "dataset_verification" in summary:
                    v = summary["dataset_verification"]
                    status = "✅" if v.get("verified") else "❌"
                    details = [d for d in v.get("details", []) if "FAILED" in d or "verified" in d]
                    print(f"      Dataset verify: {status} {details[0] if details else '—'}")

    # Local file store hints
    if tracking_uri.startswith("file://"):
        db_path = Path(tracking_uri.replace("file://", ""))
        print(f"\n  📂 Local data at: {db_path.resolve()}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="rationai.mlkit — full pipeline demo")
    parser.add_argument("--test", action="store_true", help="Run unit tests instead")
    parser.add_argument("--uri", type=str, default=None,
                        help="MLflow tracking URI (default: local file store)")
    args = parser.parse_args()

    import mlflow as _mlf
    uri = args.uri or f"file:///tmp/mlkit_demo_mlruns_{os.getpid()}"
    _mlf.set_tracking_uri(uri)
    print(f"[*] MLflow tracking URI: {uri}")

    # Verify connectivity
    try:
        _client = _mlf.MlflowClient()
        _ = _client.get_experiment_by_name("__ping__")
        print(f"[*] Connected ✅\n")
    except Exception as e:
        print(f"[!] Warning: Could not connect to MLflow server: {e}\n")

    if args.test:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from tests.test_all import main as test_main
        test_main()
        return

    sep("rationai.mlkit — End-to-End Demo")

    # Step 1 & 2: create + register datasets (required for fail_fast=True)
    step_create_datasets(n=2, wsis_per_ds=10)
    step_register_datasets()

    # Steps 3-6: component demos
    step_stream_capture()
    step_metrics()
    step_nested_metrics()
    step_sampler()

    # Steps 7-8: training runs (fail_fast=True — verification must pass)
    step_provenance()
    step_lightning()

    # Step 9: summary
    step_summary()

    sep("✅ Demo complete — all runs are in MLflow!")


if __name__ == "__main__":
    main()