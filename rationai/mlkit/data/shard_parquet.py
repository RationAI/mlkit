import logging
from pathlib import Path

import pyarrow.parquet as pq


_logger = logging.getLogger(__name__)


def shard_parquet(
    input_file: str | Path,
    output_dir: str | Path,
    rows_per_shard: int = 100_000,
    row_group_size: int = 5000,
) -> None:

    assert rows_per_shard > 0, "rows_per_shard must be grater than 0"
    assert row_group_size > 0, "row_group_size must be grater than 0"
    assert rows_per_shard % row_group_size > 0, (
        "rows_per_shard must be divisible by row_group_size"
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True)

    with pq.ParquetFile(input_file) as parquet_file:
        _logger.info(f"Total rows in source: {parquet_file.metadata.num_rows}")

        shard_idx = 0
        current_shard_rows = 0
        writer = None

        try:
            for batch in parquet_file.iter_batches(batch_size=row_group_size):
                if writer is None:
                    out_path = output_dir / f"shard_{shard_idx:05d}.parquet"
                    writer = pq.ParquetWriter(out_path, batch.schema)

                writer.write_batch(batch)
                current_shard_rows += batch.num_rows

                if current_shard_rows >= rows_per_shard:
                    writer.close()
                    writer = None
                    _logger.info(f"Finished writing shard {shard_idx:05d}")
                    shard_idx += 1
                    current_shard_rows = 0

            if writer is not None:
                _logger.info(f"Finished writing final shard {shard_idx:05d}")
        finally:
            if writer is not None:
                writer.close()

    _logger.info("Sharding complete!")
