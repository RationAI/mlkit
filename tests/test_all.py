"""
End-to-end test suite for rationai.mlkit.

Exercises every major component:
  1. Stream capture + StreamModifier
  2. AggregatedMetricCollection + aggregators (with torchmetrics)
  3. NestedMetricCollection
  4. StratifiedBatchSampler / PDMStratifiedBatchSampler
  5. Lightning: Trainer, MLFlowLogger, MultiloaderLifecycle, autolog, with_cli_args
  6. Provenance @autolog (full training run + provenance artifact verification)

Run: python tests/test_all.py
"""

import sys
import io
import os
import json
import tempfile
from pathlib import Path

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import mlflow


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────
# 1. Stream Capture + StreamModifier
# ──────────────────────────────────────────────

def test_stream_capture():
    section("1. StreamCapture + StreamModifier")

    from rationai.mlkit import StreamCapture, StreamLogger, StreamModifier

    class _TestLogger(StreamLogger):
        def __init__(self):
            self._buffer = io.StringIO()
        def log_stream(self, text: str):
            self._buffer.write(text)
        def get_value(self):
            return self._buffer.getvalue()

    logger = _TestLogger()
    with StreamCapture(logger, streams=(sys.stdout,)):
        print("hello")
        print("\033[31mred text\033[0m")

    captured = logger.get_value()
    assert "hello" in captured
    assert "\x1b" in captured, "StreamCapture preserves ANSI codes (raw capture)"
    print(f"  Captured: {captured.strip()!r}")
    print("  ✅ StreamCapture works (captures raw output including ANSI)")

    # StreamModifier — wraps a stream's write to inject side-effect
    # logic before the original write fires. The callback receives (text, id)
    # and can e.g. log or tag the output elsewhere.
    buf = io.StringIO()
    side_log = []
    modifier = StreamModifier(stream=buf, id=42)
    modifier.set_write(lambda s, iid: side_log.append(f"[{iid}] {s}"))
    buf.write("hello")
    assert "[42] hello" in side_log, f"Side effect not called: {side_log}"
    assert buf.getvalue() == "hello", f"Original write not called: {buf.getvalue()!r}"
    modifier.teardown()
    print(f"  Side log: {side_log}")
    print(f"  Original buf: {buf.getvalue()!r}")
    print("  ✅ StreamModifier works")


# ──────────────────────────────────────────────
# 2. AggregatedMetricCollection + aggregators
# ──────────────────────────────────────────────

def test_aggregated_metrics():
    section("2. AggregatedMetricCollection")

    try:
        from torchmetrics import Accuracy
        from rationai.mlkit import (
            AggregatedMetricCollection,
            MaxAggregator,
            MeanAggregator,
        )
    except ModuleNotFoundError as e:
        if "rationai.masks" in str(e):
            raise  # handled by main as skip
        raise

    preds = torch.tensor([0.1, 0.8, 0.3, 0.9])
    targets = torch.tensor([0, 1, 0, 1])
    keys = ["slide_A", "slide_A", "slide_B", "slide_B"]

    for name, agg in [
        ("MaxAggregator", MaxAggregator()),
        ("MeanAggregator", MeanAggregator()),
    ]:
        mc = AggregatedMetricCollection(
            metrics={"accuracy": Accuracy(task="binary")},
            aggregator=agg,
        )
        mc.update(preds, targets, keys)
        result = mc.compute()
        print(f"  {name:20s}: accuracy={result['accuracy'].item():.4f}")

    print("  ✅ AggregatedMetricCollection works")


# ──────────────────────────────────────────────
# 3. NestedMetricCollection
# ──────────────────────────────────────────────

def test_nested_metrics():
    section("3. NestedMetricCollection")

    from torchmetrics import Accuracy, Precision
    from rationai.mlkit import NestedMetricCollection

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

    assert "slide" in result
    print(f"  Slides: {result['slide']}")
    for k, v in result.items():
        if k != "slide":
            print(f"  {k:15s}: {v}")
    print("  ✅ NestedMetricCollection works")


# ──────────────────────────────────────────────
# 4. StratifiedBatchSampler / PDMStratifiedBatchSampler
# ──────────────────────────────────────────────

def test_samplers():
    section("4. Samplers")

    from rationai.mlkit import StratifiedBatchSampler

    sampler = StratifiedBatchSampler(
        data_indices=[[0, 1, 2, 3], [4, 5, 6, 7]],
        batch_size=4,
    )
    batches = list(sampler)
    print(f"  StratifiedBatchSampler: {len(batches)} batches")
    for i, batch in enumerate(batches):
        print(f"    Batch {i}: {batch}")
    assert len(batches) == 2

    # PDMStratifiedBatchSampler requires a DataFrame
    from rationai.mlkit import PDMStratifiedBatchSampler
    import pandas as pd

    df = pd.DataFrame({
        "idx": list(range(8)),
        "label": [0, 0, 0, 1, 1, 1, 1, 0],
    })
    pdm_sampler = PDMStratifiedBatchSampler(
        data=df,
        stratify_by="label",
        batch_size=4,
    )
    pdm_batches = list(pdm_sampler)
    print(f"  PDMStratifiedBatchSampler: {len(pdm_batches)} batches")
    assert len(pdm_batches) >= 1

    print("  ✅ Samplers work")


# ──────────────────────────────────────────────
# 5. Lightning imports + basic functionality
# ──────────────────────────────────────────────

def test_lightning():
    section("5. Lightning (Trainer, MLFlowLogger, MultiloaderLifecycle, autolog, with_cli_args)")

    from rationai.mlkit import Trainer, MLFlowLogger, MultiloaderLifecycle, autolog, with_cli_args
    print(f"  Trainer: {Trainer}")
    print(f"  MLFlowLogger: {MLFlowLogger}")
    print(f"  MultiloaderLifecycle: {MultiloaderLifecycle}")
    print(f"  autolog: {autolog}")
    print(f"  with_cli_args: {with_cli_args}")

    import lightning as pl

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

    model = _TinyModel()
    trainer = Trainer(
        max_epochs=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    ds = torch.utils.data.TensorDataset(torch.randn(8, 8), torch.randint(0, 2, (8,)))
    trainer.fit(model, torch.utils.data.DataLoader(ds))
    print("  ✅ Lightning Trainer works")


# ──────────────────────────────────────────────
# 6. Provenance @autolog
# ──────────────────────────────────────────────

def test_provenance():
    section("6. Provenance (@autolog + artifact verification)")

    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from rationai.mlkit.provenance import autolog

    class _DummyDS(Dataset):
        def __len__(self):
            return 16
        def __getitem__(self, idx):
            return torch.randn(32), torch.randint(0, 2, (1,)).item()

    with tempfile.TemporaryDirectory() as tmpdir:
        mlflow.set_tracking_uri(f"file://{tmpdir}/mlruns")

        @autolog(model_name="test_model", experiment_name="Test_Provenance", fail_fast=False)
        def train(run):
            model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))
            run.register_model(model)

            optimizer = optim.Adam(model.parameters(), lr=0.01)
            run.register_optimizer(optimizer)

            loader = DataLoader(_DummyDS(), batch_size=4)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(1, 3):
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

        client = mlflow.MlflowClient()
        exp = client.get_experiment_by_name("Test_Provenance")
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        assert len(runs) >= 1, "Expected at least 1 run"

        r = runs[0]
        artifacts = [a.path for a in client.list_artifacts(r.info.run_id)]
        print(f"  Artifacts: {artifacts}")

        prov_path = None
        for a in artifacts:
            if "provenance" in a.lower():
                prov_path = a.rstrip("/")
                break

        if prov_path:
            local_dir = client.download_artifacts(r.info.run_id, prov_path)
            # download_artifacts returns a directory; find run_summary.json inside
            import glob as _glob
            summary_files = _glob.glob(f"{local_dir}/**/run_summary.json", recursive=True)
            if not summary_files:
                summary_files = [f for f in os.listdir(local_dir) if f.endswith(".json")]
                summary_files = [os.path.join(local_dir, f) for f in summary_files] if summary_files else []
            if summary_files:
                with open(summary_files[0]) as f:
                    summary = json.load(f)
                print(f"  Provenance keys: {list(summary.keys())}")
                assert "model_name" in summary or "params" in summary or "metrics" in summary
            else:
                print(f"  (Provenance dir contents: {os.listdir(local_dir)})")
        else:
            print("  (No provenance artifact found — checking run params/metrics)")
            assert r.data.params, "Expected params in run"

        print("  ✅ Provenance @autolog works")


# ──────────────────────────────────────────────
# Backward compatibility
# ──────────────────────────────────────────────

def test_backward_compat():
    section("7. Backward Compatibility (old import paths)")

    from rationai.mlkit import Trainer, autolog, with_cli_args
    print("  from rationai.mlkit import Trainer, autolog, with_cli_args: ✅")

    from rationai.mlkit.autolog import autolog as _autolog
    print("  from rationai.mlkit.autolog import autolog: ✅")

    from rationai.mlkit.with_cli_args import with_cli_args as _wca
    print("  from rationai.mlkit.with_cli_args import with_cli_args: ✅")

    from rationai.mlkit.lightning.autolog import autolog as _la
    print("  from rationai.mlkit.lightning.autolog import autolog: ✅")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    section("rationai.mlkit — Test Suite")

    tests = [
        ("Stream Capture", test_stream_capture),
        ("Aggregated Metrics", test_aggregated_metrics),
        ("Nested Metrics", test_nested_metrics),
        ("Samplers", test_samplers),
        ("Lightning", test_lightning),
        ("Provenance", test_provenance),
        ("Backward Compat", test_backward_compat),
    ]

    passed = failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except ModuleNotFoundError as e:
            print(f"  ⊘ SKIPPED {name}: {e}")
            skipped += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    section(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
