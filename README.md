# rationai.mlkit — Unified ML Provenance & Metrics Toolkit

Automatic PROV-O-aware experiment tracking for pathology image classification,
with Lightning/Hydra integration and pathology-specific metric layers.

All metadata (user, dataset, hardware, docker, git, environment, model architecture,
optimizer/scheduler settings, console output) is captured automatically via the
`@autolog` decorator and `ProvenanceCallback` — the user only writes their training loop.

---

## Setup

```bash
uv venv
source .venv/bin/activate
uv sync
```

Start the MLflow server:

```bash
mlflow ui --host 127.0.0.1 --port 5000    # → http://localhost:5000
```

---

## Quick start

### 1. Register users and dataset (one-time setup)

Before training, register researchers and datasets so provenance can reference them:

```bash
# Edit example_provenance_setup.py with your own data, then run:
python example_provenance_setup.py
```

This creates runs in the `User_Registry` and `Dataset_Registry` MLflow experiments.

### 2. Run a training experiment

Use `@autolog` + `ProvenanceCallback` in a Hydra-based training script:

```bash
python example_provenance_train.py
```

Check the MLflow UI at http://localhost:5000 to see the full provenance graph.

See [example_provenance_setup.py](example_provenance_setup.py) and
[example_provenance_train.py](example_provenance_train.py) for complete working examples.

---

## API reference

### User registration

Register a researcher into the `User_Registry` experiment:

```python
from rationai.mlkit.provenance import register_new_user

register_new_user(
    username="researcher_01",
    real_name="Jane Doe",
    email="jane.doe@example.com",
    organization="Example Org",
    lead_name="John Smith",
    lead_email="john.smith@example.com",
)
```

### Dataset registration

Register a dataset (requires `manifest.csv` in the dataset directory):

```python
from rationai.mlkit.provenance import register_dataset

run_id = register_dataset(
    dataset_dir="data/cohorts/pato_01",
    dataset_name="pato_cohort_01",
    version="2.0",
)
print(f"Registered as {run_id}")
```

Verify a dataset's file integrity:

```python
from rationai.mlkit.provenance import verify_dataset

result = verify_dataset(manifest_path="data/cohorts/pato_01/manifest.csv")
print(result["verified"])  # True if all files match
```

### Provenance — `@autolog` decorator (Hydra training scripts)

Full auto-capture for Hydra-based training runs:

```python
import hydra
from omegaconf import DictConfig
from rationai.mlkit import Trainer, autolog
from rationai.mlkit.lightning.loggers.mlflow import MLFlowLogger
from rationai.mlkit.lightning.callbacks import ProvenanceCallback

@hydra.main(config_path=".", config_name="train_cfg", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    model = MyLightningModule()
    data = MyDataModule(batch_size=config.batch_size)

    trainer = Trainer(
        max_epochs=config.epochs,
        logger=logger,
        callbacks=[
            ProvenanceCallback(model_name="my_model_v1")
        ],
    )
    trainer.fit(model, datamodule=data)
```

| Auto-captured | Details |
|---|---|
| **User** | Resolved from `MLFLOW_USER` env → git config → linked to `User_Registry` run |
| **Dataset** | Latest `Dataset_Registry` run, file sizes verified |
| **Train/test split** | Manifest auto-discovered, CSVs saved as artifacts |
| **Model architecture** | Class name, param counts, per-layer summary |
| **Optimizer** | Type, lr, momentum, weight_decay, … |
| **Scheduler** | Type, step_size, gamma, milestones, … |
| **Hardware** | GPU/CPU/RAM/OS/Python version |
| **Docker** | Container ID, image name + hash (if in container) |
| **Git** | Commit, branch, remote URL |
| **Environment** | Frozen `requirements.txt` |
| **Console output** | stdout/stderr → `logs/console.log` artifact (ANSI-aware) |
| **PROV document** | OpenProvenance JSON → `provenance/prov.json` artifact |

### Lightning callbacks

#### ProvenanceCallback

Drop-in callback that captures full PROV-O provenance for every training run:

```python
from rationai.mlkit.lightning.callbacks import ProvenanceCallback

callback = ProvenanceCallback(
    model_name="resnet_v1",
    experiment_name="Training_Pipeline",
)

trainer = Trainer(callbacks=[callback], ...)
```

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `None` | Model identifier (used in artifact paths) |
| `experiment_name` | `"Training_Pipeline"` | MLflow experiment name |
| `manifest_path` | auto-discover | Path to `manifest.csv` |
| `data_root` | auto-discover | Root directory of dataset |
| `test_size` | `0.2` | Test split fraction |
| `random_state` | `42` | Random seed for split |
| `fail_fast` | `True` | Stop training on verification failure |
| `strict` | `False` | Require all provenance fields |
| `register_model` | `True` | Register model architecture to MLflow |
| `register_optimizer` | `True` | Register optimizer config to MLflow |
| `register_scheduler` | `True` | Register scheduler config to MLflow |

#### EnvironmentCallback

Captures environment metadata (git, hardware, docker, environment freeze):

```python
from rationai.mlkit.lightning.callbacks import EnvironmentCallback

callback = EnvironmentCallback(
    skip_hardware=False,   # capture GPU/CPU/RAM info
    snapshot_env=True,     # freeze requirements.txt
)
```

#### DatasetVerificationCallback

Verifies dataset integrity and optionally performs a stratified split:

```python
from rationai.mlkit.lightning.callbacks import DatasetVerificationCallback

callback = DatasetVerificationCallback(
    test_size=0.2,       # 0.0 to skip split
    random_state=42,
    fail_fast=True,      # stop training if verification fails
)
```

### Metrics

#### AggregatedMetricCollection — tile → slide aggregation

Group tile-level predictions by slide and compute metrics at the slide level:

```python
from torchmetrics import Accuracy
from rationai.mlkit import (
    AggregatedMetricCollection,
    MaxAggregator,
    MeanAggregator,
)

agg = AggregatedMetricCollection(
    metrics={"accuracy": Accuracy(task="binary")},
    aggregator=MaxAggregator(),
)

for preds, targets, slide_ids in val_loader:
    agg.update(preds, targets, keys=slide_ids)

results = agg.compute()
# → {"key": ["slide1", "slide2"], "accuracy": [0.95, 0.87]}
```

Available aggregators: `MaxAggregator`, `MeanAggregator`, `TopKAggregator`, `MeanPoolMaxAggregator`.

#### NestedMetricCollection — per-slide multiclass metrics

Compute multiple torchmetrics per slide with class-level breakdowns:

```python
from rationai.mlkit import NestedMetricCollection
from torchmetrics import Accuracy, Precision

metrics = NestedMetricCollection(
    metrics={
        "accuracy": Accuracy(task="multiclass", num_classes=3),
        "precision": Precision(task="multiclass", num_classes=3, average=None),
    },
    key_name="slide",
    class_names=["benign", "low_grade", "high_grade"],
)

metrics.update(preds, targets, keys=slide_ids)
result = metrics.compute()
```

### Stream capture — ANSI-aware stdout/stderr logging

Captures console output (including progress bars and ANSI color codes)
without corrupting the log:

```python
from rationai.mlkit import StreamCapture

with StreamCapture(stream="stdout") as capture:
    print("Hello!")
    print("\033[92mGreen text\033[0m")  # ANSI color

text = capture.get_text()       # Raw captured text
clean = capture.get_clean_text() # ANSI codes stripped
```

### Data utilities

#### StratifiedBatchSampler

Balanced class sampling across batches:

```python
from rationai.mlkit import StratifiedBatchSampler

sampler = StratifiedBatchSampler(
    data_indices=[[0, 1, 2, 3], [4, 5, 6, 7]],  # per-class indices
    batch_size=4,
)
for batch in sampler:
    train(model, batch)
```

#### MetaTiledSlides

Load tile data from parquet or MLflow artifact URIs:

```python
from rationai.mlkit import MetaTiledSlides

dataset = MetaTiledSlides(
    manifest_uri="s3://bucket/data/manifest.parquet",
    tile_size=256,
)
```

### Lightning integration

| Component | Import | Purpose |
|---|---|---|
| `Trainer` | `from rationai.mlkit import Trainer` | Lightning Trainer with MLflow checkpoint sync |
| `MLFlowLogger` | `from rationai.mlkit import MLFlowLogger` | Logger with git tags, stream capture, checkpoint sync |
| `MultiloaderLifecycle` | `from rationai.mlkit import MultiloaderLifecycle` | Per-dataloader callback hooks |
| `ProvenanceCallback` | `from rationai.mlkit.lightning.callbacks import ProvenanceCallback` | Full PROV-O provenance capture |
| `EnvironmentCallback` | `from rationai.mlkit.lightning.callbacks import EnvironmentCallback` | Environment metadata capture |
| `DatasetVerificationCallback` | `from rationai.mlkit.lightning.callbacks import DatasetVerificationCallback` | Dataset verification + split |
| `with_cli_args` | `from rationai.mlkit import with_cli_args` | Programmatic config injection (Hydra) |

---

## Project structure

```text
.
├── example_provenance_setup.py        # Setup: register users & dataset
├── example_provenance_train.py        # Training with @autolog + ProvenanceCallback
├── example_provenance_train_cfg.yaml  # Hydra config for the training example
├── pyproject.toml                     # Project metadata + deps
├── test_data/                         # Dummy datasets (gitignored)
└── rationai/
    └── mlkit/
        ├── __init__.py                # Package exports (lazy loading)
        ├── autolog.py                 # @autolog decorator for Hydra scripts
        ├── with_cli_args.py           # Programmatic config injection
        ├── provenance/                # Dataset & user registration
        │   ├── __init__.py
        │   ├── register_dataset.py    # register_dataset, verify_dataset
        │   └── register_user.py       # register_new_user
        ├── stream/                    # ANSI-aware console capture
        │   ├── stream_capture.py
        │   ├── stream_logger.py
        │   └── stream_modifier.py
        ├── metrics/                   # Slide-level metric aggregation
        │   ├── aggregated_metric_collection.py
        │   ├── nested_metric_collection.py
        │   ├── aggregators.py
        │   └── lazy_metric_dict.py
        ├── data/                      # Data utilities
        │   ├── shard_parquet.py
        │   ├── samplers/
        │   │   └── stratified_batch_sampler.py
        │   └── datasets/
        │       ├── meta_tiled_slides.py
        │       ├── openslide_tiles_dataset.py
        │       └── slides_tiles_loader.py
        └── lightning/                 # Lightning + Hydra integration
            ├── trainer.py
            ├── with_cli_args.py
            ├── callbacks/
            │   ├── provenance.py      # ProvenanceCallback
            │   ├── environment.py     # EnvironmentCallback
            │   ├── dataset_verification.py  # DatasetVerificationCallback
            │   └── multiloader_lifecycle.py
            └── loggers/
                └── mlflow.py          # MLFlowLogger (checkpoint sync, git tags)
```

---

## MLflow experiments

| Experiment | Purpose |
|---|---|
| `User_Registry` | Stores user identity runs (username, real name, org) |
| `Dataset_Registry` | Stores dataset manifest runs with file provenance |
| `Training_Pipeline` | Training runs with full auto-captured provenance |

Cross-run tags (`user_run_id`, `dataset_run_id`) on training runs make
PROV graph reconstruction machine-readable.

---

## PROV document (W3C PROV-O)

Each training run emits a self-contained OpenProvenance JSON document at
`provenance/prov.json` (MLflow artifact). This document is compatible with
the `prov_mlflow` Java tool and
follows the W3C PROV-O standard.

### Structure

The document uses a bundle wrapper (`{"bundle": {"storage:<run_id>": {...}}}`)
with 10 namespace prefixes and 7 sections:

| Section | Purpose |
|---|---|
| `prefix` | Namespace URIs (`gen:`, `schema:`, `cpm:`, `prov:`, `sosa:`, …) |
| `entity` | Input data (dataset/WSI as `sosa:Sample`), metadata bundle (`cpm:BundleMetadata`) |
| `activity` | Training run (`schema:Action`) with hyperparameters, hardware, git commit; CPM wrapper (`cpm:mainActivity`) |
| `agent` | Researcher (`schema:Person`) with name, email, affiliation |
| `used` | Run activity → dataset entity |
| `wasAssociatedWith` | Run activity → agent |
| `wasGeneratedBy` | Metadata bundle ← run activity |

### PROV elements

| Element | ID Pattern | Type | Content |
|---|---|---|---|
| Agent | `gen:user_<username>` | `schema:Person` | Name, email, affiliation |
| Dataset Entity | `gen:dataset_<id>` | `sosa:Sample` | Input data (or WSI if path available) |
| Run Activity | `gen:run_<run_id>` | `schema:Action` | Hyperparams, hardware, git, optimizer/scheduler settings |
| Meta Bundle | `meta:<run_id>` | `cpm:BundleMetadata` | Full param/metric snapshot (model arch, layer summary, split info, final metrics) |
| Main Activity | `blank:TrainingRun_<id>` | `cpm:mainActivity` | CPM wrapper linking meta bundle to run activity |

### Compatibility

The generated document matches the Java `prov_mlflow`
output format:

- Bundle-wrapped JSON structure
- Qualified type annotations (`{"type": "prov:QUALIFIED_NAME", "$": "schema:Person"}`)
- Array-wrapped property values (Java convention)
- Blank node IDs for relationship references (`_:n0`, `_:n1`, …)
- CPM metadata entity with hardware, model, and metric details

### Accessing the document

```bash
# Via MLflow UI → Artifacts tab → provenance/prov.json

# Via Python API
import mlflow, json
mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.tracking.MlflowClient()
run = client.search_runs("<experiment_id>")[-1]
prov_path = client.download_artifacts(run.info.run_id, "provenance/prov.json")
with open(prov_path) as f:
    prov = json.load(f)
```
