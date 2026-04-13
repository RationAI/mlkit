import os

import pyarrow.parquet as pq


def shard_parquet(input_file: str, output_dir: str, rows_per_shard: int = 100_000, row_group_size: int = 5000) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the file metadata (does not load data into RAM)
    parquet_file = pq.ParquetFile(input_file)
    print(f"Total rows in source: {parquet_file.metadata.num_rows}")
    
    shard_idx = 0
    current_shard_rows = 0
    writer = None
    
    # Stream the file in tiny, memory-safe batches
    for batch in parquet_file.iter_batches(batch_size=row_group_size):
        
        # Initialize a new file writer if we don't have one open
        if writer is None:
            out_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.parquet")
            # We enforce the smaller row_group_size here for the new files
            writer = pq.ParquetWriter(out_path, batch.schema)
        
        # Write the batch to the current shard
        writer.write_batch(batch, row_group_size=row_group_size)
        current_shard_rows += batch.num_rows
        
        # If we hit our row limit for this shard, close it and increment
        if current_shard_rows >= rows_per_shard:
            writer.close()
            writer = None
            print(f"Finished writing shard {shard_idx:05d}")
            shard_idx += 1
            current_shard_rows = 0

    # Clean up the last writer
    if writer is not None:
        writer.close()
        print(f"Finished writing final shard {shard_idx:05d}")
        
    print("Sharding complete!")