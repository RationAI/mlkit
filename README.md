# rationai.mlkit — Unified ML Provenance & Metrics Toolkit

Automatic PROV-O-aware experiment tracking for pathology image classification,
with Lightning/Hydra integration and pathology-specific metric layers.

All metadata (user, dataset, hardware, docker, git, environment, model architecture,
optimizer/scheduler settings, console output) is captured automatically via the
`@autolog` decorator — the user only writes their training loop.

---

## Setup

```bash
uv venv
source .venv/bin/activate
uv sync
```

Start the MLflow server:

```bash
mlflow ui --host 0.0.0.0 --port 5000    # → http://localhost:5000
```

---

## Quick start

### 1. Run the demo

The easiest way to see everything in action:

```bash
# Full pipeline — uploads to localhost:5000
python demo.py

# Push to a different MLflow server
python demo.py --uri http://your-server:5000

# Run unit tests instead
python demo.py --test
```

The demo exercises all components:

| Step | Feature | What it shows |
|---|---|---|
| 1 | Dummy data creation | `test_data/dummy_dataset_*` with manifests |
| 2 | **StreamCapture** | ANSI-aware stdout/stderr capture |
| 3 | **AggregatedMetricCollection** | Tile → slide metric aggregation |
| 4 | **NestedMetricCollection** | Per-slide multiclass metrics |
| 5 | **StratifiedBatchSampler** | Balanced class batches |
| 6 | **Provenance** (`@autolog`) | Full training run with auto-captured provenance |
| 7 | **Lightning** (Trainer + MLFlowLogger) | Lightning training with full provenance tracking |

Both steps 6 and 7 upload to MLflow with identical provenance depth:
model params, GPU/CPU info, optimizer config, train/test split stats,
environment freeze, console logs, and the PROV-O document.

### 2. Create dummy data

Generate test datasets (no shell script needed):

```bash
# Default: 2 datasets × 50 WSIs each
python dummy_dataset_create.py

# Custom: 3 datasets with 100 WSIs each
python dummy_dataset_create.py --datasets 3 --wsis-per-dataset 100

# Add more without deleting existing ones
python dummy_dataset_create.py -d 1 -w 30 --no-clean
```

This creates the following structure under `data/`:

```
data/
├── dummy_dataset_1/
│   ├── manifest.csv          # patient_id, wsi_path (relative), cancer
│   └── wsis/
│       ├── PAT_001.tiff
│       ├── PAT_002.tiff
│       └── ...
├── dummy_dataset_2/
│   ├── manifest.csv
│   └── wsis/
│       └── ...
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--datasets N` / `-d` | `2` | Number of dataset folders |
| `--wsis-per-dataset N` / `-w` | `50` | WSIs per dataset |
| `--data-dir DIR` | `data/` | Parent directory |
| `--seed N` | `42` | Reproducibility seed |
| `--no-clean` | off | Keep existing datasets, append new ones |
| `--img-size N` | `128` | Pixel size of dummy TIFF images |

### 3. Register a user

Edit the variables in `user_to_mlflow.py` and run:

```bash
python user_to_mlflow.py
```

This creates a run in the **User_Registry** experiment with your identity tags.

### 4. Run a training experiment

Use the `@autolog` decorator from `rationai.mlkit.provenance` in your training script, then:

```bash
python your_experiment.py
```

See the [API reference](#provenance--autolog-decorator) below for details.

---

## API reference

### Provenance — `@autolog` decorator

Full auto-capture for plain PyTorch training runs:

```python
from rationai.mlkit.provenance import autolog

@autolog(model_name="my_model_v1", experiment_name="My_Experiment")
def train(run):
    model = build_model()
    run.register_model(model)

    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    run.register_optimizer(optimizer)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    run.register_scheduler(scheduler)

    for epoch in range(50):
        loss = train_epoch(model, loader, optimizer)
        run.log_metrics({"train_loss": loss}, step=epoch)

    run.save_model(model)

if __name__ == "__main__":
    train()
```

| Auto-captured | Details |
|---|---|
| **User** | Resolved from git config → linked to `User_Registry` run |
| **Dataset** | Latest `Dataset_Registry` run |
| **Train/test split** | Manifest auto-discovered, CSVs saved as artifacts |
| **Model architecture** | Class name, param counts, per-layer summary |
| **Optimizer** | Type, lr, momentum, weight_decay, … |
| **Scheduler** | Type, step_size, gamma, milestones, … |
| **Hardware** | GPU/CPU/RAM/OS/Python version |
| **Docker** | Container ID, image name + hash |
| **Git** | Commit, branch, remote URL |
| **Environment** | Frozen `requirements.txt` + `pyproject.toml` / `uv.lock` |
| **Console output** | stdout/stderr → `logs/console.log` artifact (ANSI-aware) |
| **PROV document** | OpenProvenance JSON → `provenance/prov.json` artifact |

### Provenance + Lightning

Wrap Lightning training in `@autolog` and pass the active run to `MLFlowLogger`:

```python
import mlflow
from rationai.mlkit import Trainer, MLFlowLogger
from rationai.mlkit.provenance import autolog

@autolog(model_name="my_lightning_model", experiment_name="My_Experiment")
def train(run):
    model = MyLightningModule()
    run.register_model(model)
    run.register_optimizer(model.configure_optimizers())

    # Reuse the @autolog run so Lightning logs to the same provenance-tracked run
    logger = MLFlowLogger(experiment_name="My_Experiment", run_id=mlflow.active_run().info.run_id)

    trainer = Trainer(logger=logger, max_epochs=50)
    trainer.fit(model, train_loader)

    run.save_model(model)
```

This gives you the same full provenance depth as plain PyTorch — GPU info,
model architecture, optimizer config, environment freeze, PROV document —
plus Lightning's native metric logging via `self.log()`.

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
| `lightning_autolog` | `from rationai.mlkit.lightning import autolog` | Lightning-specific autolog decorator |
| `with_cli_args` | `from rationai.mlkit.lightning import with_cli_args` | Programmatic config injection (Hydra) |

---

## Project structure

```
.
├── demo.py                      # End-to-end demo (all components)
├── dummy_dataset_create.py      # Dummy data generator (CLI)
├── user_to_mlflow.py            # User registration script
├── pyproject.toml               # Project metadata + deps
├── tests/
│   └── test_all.py              # Unit test suite
├── test_data/                   # Dummy datasets (gitignored)
│   ├── dummy_dataset_1/
│   │   ├── manifest.csv
│   │   └── wsis/
│   └── ...
└── rationai/
    └── mlkit/
        ├── __init__.py          # Package exports (lazy Lightning import)
        ├── autolog.py           # Re-exports lightning.autolog
        ├── with_cli_args.py     # Re-exports lightning.with_cli_args
        ├── stream/              # ANSI-aware console capture
        │   ├── stream_capture.py
        │   ├── stream_logger.py
        │   └── stream_modifier.py
        ├── metrics/             # Slide-level metric aggregation
        │   ├── aggregated_metric_collection.py
        │   ├── nested_metric_collection.py
        │   ├── aggregators.py
        │   └── lazy_metric_dict.py
        ├── data/                # Data utilities
        │   ├── samplers/
        │   │   └── stratified_batch_sampler.py
        │   └── datasets/
        │       ├── meta_tiled_slides.py
        │       └── openslide_tiles_dataset.py
        └── lightning/           # Lightning + Hydra integration
            ├── autolog.py
            ├── trainer.py
            ├── with_cli_args.py
            ├── callbacks/
            │   └── multiloader_lifecycle.py
            └── loggers/
                └── mlflow.py    # MLFlowLogger (checkpoint sync, git tags)
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
the [prov_mlflow](https://github.com/jiribuchta/prov_mlflow) Java tool and
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

The generated document matches the Java [`prov_mlflow`](https://github.com/jiribuchta/prov_mlflow)
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
