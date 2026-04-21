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
    """Represents a lazy-loaded Parquet dataset with MLflow Tracking."""

    def __init__(
        self,
        path: str,
        source: DatasetSource,
        target_col: str | None = None,
        name: str | None = None,
        digest: str | None = None,
    ):
        """Hety.

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
        
        # Lazily load the dataset metadata without reading the data into memory
        self._ds = ds.dataset(self._path, format="parquet")
        
        super().__init__(source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """Computes a fast digest for the dataset based on schema and file paths."""
        hasher = hashlib.md5()
        
        # Hash the schema structure
        hasher.update(str(self._ds.schema).encode("utf-8"))
        
        # Hash the sorted file paths to detect added/removed chunks
        for file in sorted(self._ds.files):
            hasher.update(file.encode("utf-8"))
            
        return hasher.hexdigest()

    def to_dict(self) -> dict[str, str]:
        """Create config dictionary for the dataset."""
        config = super().to_dict()
        config.update({
            "schema": json.dumps({"mlflow_colspec": self.schema.to_dict()}),
            "profile": json.dumps(self.profile),
        })
        return config

    @property
    def source(self) -> DatasetSource:
        """The source of the dataset."""
        return self._source

    @property
    def dataset(self) -> ds.Dataset:
        """The underlying pyarrow Dataset object."""
        return self._ds

    @property
    def target_col(self) -> str | None:
        """The name of the target column, if specified."""
        return self._target_col

    @property
    def profile(self) -> Any:
        """A profile of the dataset metadata.

        Reads Parquet footers to instantly get row counts and structural metadata 
        without loading the actual data blocks into memory.
        """
        total_rows = 0
        
        # Iterate over file fragments to read metadata directly from Parquet footers
        for fragment in self._ds.get_fragments():
            # ParquetFileFragment natively exposes 'metadata'
            if hasattr(fragment, "metadata") and fragment.metadata is not None:
                chunk_rows = fragment.metadata.num_rows
            else:
                chunk_rows = fragment.count_rows()
                
            total_rows += chunk_rows

        return {
            "num_files": len(self._ds.files),
            "total_rows": total_rows,
            "num_columns": len(self._ds.schema.names),
            "backend_format": "parquet",
        }

    @cached_property
    def schema(self) -> Schema:
        """MLflow Schema representing the dataset features."""
        try:
            # Fetch an empty Pandas dataframe from the schema to utilize MLflow's built-in 
            # inference securely and correctly map PyArrow types to MLflow types.
            empty_df = self._ds.head(0).to_pandas()
            inferred_schema = _infer_schema(empty_df)
            return inferred_schema
        except Exception as e:
            _logger.warning(f"Failed to infer schema for Parquet dataset. Exception: {e}")
            return Schema([])


def from_parquet(
    path: str,
    source: str | DatasetSource | None = None,
    target_col: str | None = None,
    name: str | None = None,
    digest: str | None = None,
) -> ParquetDataset:
    """Constructs a ParquetDataset object from a single Parquet file or directory.
    
    Args:
        path: Path to the Parquet file or directory of Parquet chunks.
        source: The source from which the dataset was derived. 
        target_col: Optional column name for the target.
        name: The name of the dataset.
        digest: The dataset digest (hash). If unspecified, a metadata digest is computed.

    Example:
    
    .. code-block:: python
        import mlflow
        
        # Works for both a single file and a directory of chunks
        dataset = from_parquet(
            path="/path/to/massive_dataset.parquet", 
            target_col="label"
        )
        mlflow.log_input(dataset, context="training")
    """
    from mlflow.data.code_dataset_source import CodeDatasetSource
    from mlflow.data.dataset_source_registry import resolve_dataset_source
    from mlflow.tracking.context import registry

    if source is not None:
        if isinstance(source, DatasetSource):
            resolved_source = source
        else:
            resolved_source = resolve_dataset_source(source)
    else:
        context_tags = registry.resolve_tags()
        resolved_source = CodeDatasetSource(tags=context_tags)
        
    return ParquetDataset(
        path=path, 
        source=resolved_source, 
        target_col=target_col, 
        name=name, 
        digest=digest
    )