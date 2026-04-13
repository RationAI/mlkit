import logging
from pathlib import Path

import pyarrow.parquet as pq


# Initialize a module-level logger
_logger = logging.getLogger(__name__)


def shard_parquet(
    input_file: str | Path,
    output_dir: str | Path,
    rows_per_shard: int = 100_000,
    row_group_size: int = 5000,
) -> None:
    """Splits a large Parquet file into smaller Parquet files (shards).

    This function reads a single Parquet file in memory-efficient batches and writes
    it out into multiple smaller files. Each output file will contain exactly
    `rows_per_shard` rows, except potentially the final shard.

    Args:
        input_file (str | Path): The path to the source Parquet file.
        output_dir (str | Path): The directory where the output shards will be saved.
        rows_per_shard (int, optional): The target number of rows per shard.
            Defaults to 100,000.
        row_group_size (int, optional): The number of rows to read/write per batch.
            Defaults to 5,000.

    Raises:
        AssertionError: If `rows_per_shard` or `row_group_size` are not strictly positive,
            or if `rows_per_shard` is not perfectly divisible by `row_group_size`.
    """
    # --- Input Validation ---
    assert rows_per_shard > 0, "rows_per_shard must be greater than 0"
    assert row_group_size > 0, "row_group_size must be greater than 0"

    # Ensure exact chunks can be written without remainder
    assert rows_per_shard % row_group_size == 0, (
        "rows_per_shard must be divisible by row_group_size"
    )

    # --- Setup Output Directory ---
    output_dir = Path(output_dir)

    # Create the target directory and any intermediate directories if they don't exist.
    # exist_ok=True prevents crashes if you run the script multiple times.
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Read and Shard Process ---
    # Open the Parquet file as a context manager so it safely closes when done
    with pq.ParquetFile(input_file) as parquet_file:
        _logger.info(f"Total rows in source: {parquet_file.metadata.num_rows}")

        # Initialize tracking variables
        shard_idx = 0  # Tracks the current shard file number (e.g., 00000)
        current_shard_rows = (
            0  # Tracks how many rows have been written to the current shard
        )
        writer = None  # Holds the active pyarrow ParquetWriter instance

        try:
            # Iterate through the source file in memory-efficient chunks (batches)
            for batch in parquet_file.iter_batches(batch_size=row_group_size):
                # If we don't have an active writer, create a new one for the current shard
                if writer is None:
                    out_path = output_dir / f"shard_{shard_idx:05d}.parquet"
                    writer = pq.ParquetWriter(out_path, batch.schema)

                # Write the current batch and update the running row count
                writer.write_batch(batch)
                current_shard_rows += batch.num_rows

                # Check if the current shard has reached its maximum capacity
                if current_shard_rows >= rows_per_shard:
                    # Finalize the current file
                    writer.close()
                    writer = None  # Reset the writer so a new one spawns on the next loop iteration

                    _logger.info(f"Finished writing shard {shard_idx:05d}")

                    # Prepare counters for the next shard
                    shard_idx += 1
                    current_shard_rows = 0

            # After the loop ends, check if there's a partially filled final shard left open
            if writer is not None:
                _logger.info(f"Finished writing final shard {shard_idx:05d}")

        finally:
            # Ensure the active writer is properly closed even if an unexpected error occurs
            if writer is not None:
                writer.close()

    _logger.info("Sharding complete!")
