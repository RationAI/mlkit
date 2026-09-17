import hashlib
import json
import logging
from functools import cached_property
from typing import Any

import pyarrow.dataset as ds
from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.types.schema import Schema
from mlflow.types.utils import _infer_schema


_logger = logging.getLogger(__name__)


class ParquetDataset(Dataset):
    """Lazy-loaded Parquet dataset (single file or sharded directory) with MLflow tracking."""

    def __init__(
        self,
        path: str,
        source: DatasetSource,
        target_col: str | None = None,
        name: str | None = None,
        digest: str | None = None,
    ) -> None:
        """Initializes the ParquetDataset.

        Args:
            path: Local path or URI to the Parquet file or directory.
            source: The source of the parquet dataset.
            target_col: The name of the column representing the target variable. Optional.
            name: The name of the dataset. If unspecified, a name is automatically generated.
            digest: The digest (hash) of the dataset. If unspecified, a fast metadata-based
                digest is automatically computed to avoid hashing massive files.
        """
        self._path = path
        self._target_col = target_col
        self._ds = ds.dataset(self._path, format="parquet")
        super().__init__(source=source, name=name, digest=digest)

    # ── MLflow Dataset interface ──────────────────────────────────────────────

    @property
    def data_type(self) -> str:
        return "parquet"

    @property
    def source(self) -> DatasetSource:
        return self._source

    @property
    def target_col(self) -> str | None:
        return self._target_col

    @property
    def dataset(self) -> ds.Dataset:
        return self._ds

    # ── digest ────────────────────────────────────────────────────────────────

    def _compute_digest(self) -> str:
        """Fast metadata-based digest — hashes schema + sorted file paths, never reads data."""
        hasher = hashlib.md5()
        hasher.update(str(self._ds.schema).encode())
        for f in sorted(self._ds.files):
            hasher.update(f.encode())
        return hasher.hexdigest()

    # ── profile ───────────────────────────────────────────────────────────────

    @property
    def profile(self) -> dict[str, Any]:
        """Row counts and structural metadata read from Parquet footers (no data blocks loaded)."""
        total_rows = 0
        for fragment in self._ds.get_fragments():
            if hasattr(fragment, "metadata") and fragment.metadata is not None:
                total_rows += fragment.metadata.num_rows
            else:
                total_rows += fragment.count_rows()
        return {
            "num_files": len(self._ds.files),
            "total_rows": total_rows,
            "num_columns": len(self._ds.schema.names),
            "backend_format": "parquet",
        }

    # ── schema ────────────────────────────────────────────────────────────────

    @cached_property
    def schema(self) -> Schema | None:
        try:
            import pyarrow as pa
            from mlflow.types.schema import Array, ColSpec, DataType, TensorSpec

            pa_schema = self._ds.schema

            def _is_scalar(t: pa.DataType) -> bool:
                return (
                    pa.types.is_integer(t) or pa.types.is_floating(t)
                    or pa.types.is_boolean(t) or pa.types.is_string(t)
                    or pa.types.is_large_string(t) or pa.types.is_binary(t)
                    or pa.types.is_date(t) or pa.types.is_timestamp(t)
                )

            def _leaf_dtype(t: pa.DataType) -> DataType | None:
                if pa.types.is_boolean(t):
                    return DataType.boolean
                if pa.types.is_integer(t):
                    return DataType.long
                if pa.types.is_floating(t):
                    return DataType.double
                return None

            def _array_colspec(field: pa.Field) -> ColSpec | None:
                """Build a nested Array ColSpec for fixed-shape tensor/list columns.

                MLflow's Schema requires all-ColSpec or all-TensorSpec — never mixed —
                so array/tensor columns are represented as ColSpec(Array(...)) rather
                than TensorSpec, to stay homogeneous with the scalar columns below.
                """
                t = field.type
                # Tensor extension types (Ray's ArrowTensorType, PyArrow's native
                # fixed_shape_tensor) expose .shape plus a leaf element type — Ray uses
                # .scalar_type, PyArrow uses .value_type. Their .storage_type is a
                # *flattened* list (e.g. large_list<uint8>), so it can't be used to
                # recover per-dimension shape and must not be unwrapped for ndims.
                if isinstance(t, pa.ExtensionType):
                    shape = getattr(t, "shape", None)
                    leaf_type = getattr(t, "scalar_type", None) or getattr(t, "value_type", None)
                    if shape is not None and leaf_type is not None:
                        leaf = _leaf_dtype(leaf_type)
                        if leaf is not None:
                            arr: DataType | Array = leaf
                            for _ in range(len(shape)):
                                arr = Array(arr)
                            return ColSpec(arr, name=field.name)
                    t = t.storage_type  # unknown extension: fall through as a plain list
                dims = 0
                while pa.types.is_fixed_size_list(t) or pa.types.is_list(t) or pa.types.is_large_list(t):
                    dims += 1
                    t = t.value_type
                if dims == 0:
                    return None
                leaf = _leaf_dtype(t)
                if leaf is None:
                    return None
                arr = leaf
                for _ in range(dims):
                    arr = Array(arr)
                return ColSpec(arr, name=field.name)

            scalar_fields = [f for f in pa_schema if _is_scalar(f.type)]
            array_specs = [
                spec for f in pa_schema
                if not _is_scalar(f.type)
                if (spec := _array_colspec(f)) is not None
            ]

            if not scalar_fields and not array_specs:
                return None

            specs: list[ColSpec | TensorSpec] = list(array_specs)
            if scalar_fields:
                empty_table = pa.table({f.name: pa.array([], type=f.type) for f in scalar_fields})
                scalar_schema = _infer_schema(empty_table.to_pandas())
                specs = list(scalar_schema.inputs) + specs

            return Schema(specs)
        except Exception as exc:
            _logger.warning("Failed to infer schema for Parquet dataset: %s", exc)
            return None

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, str]:
        config = super().to_dict()
        if self.schema is not None:
            config["schema"] = json.dumps({"mlflow_colspec": self.schema.to_dict()})
        config["profile"] = json.dumps(self.profile)
        return config


# ── factory ───────────────────────────────────────────────────────────────────

def from_parquet(
    path: str,
    source: str | DatasetSource | None = None,
    target_col: str | None = None,
    name: str | None = None,
    digest: str | None = None,
) -> ParquetDataset:
    """Construct a ParquetDataset from a single file or a directory of shards.

    Example::

        dataset = from_parquet("/path/to/tiles/", target_col="tumor")
        mlflow.log_input(dataset, context="tiles")
    """
    from mlflow.data.code_dataset_source import CodeDatasetSource
    from mlflow.data.dataset_source_registry import resolve_dataset_source
    from mlflow.tracking.context import registry

    if source is not None:
        resolved_source = source if isinstance(source, DatasetSource) else resolve_dataset_source(source)
    else:
        resolved_source = CodeDatasetSource(tags=registry.resolve_tags())

    return ParquetDataset(path=path, source=resolved_source, target_col=target_col, name=name, digest=digest)
