import pandas as pd
from pathlib import Path

# Define input/output paths
input_file = "data/processed_train.csv"
chunks_dir = Path("data/chunks")
chunks_dir.mkdir(parents=True, exist_ok=True)

# Split into chunks of 10,000 rows
chunk_size = 10000
reader = pd.read_csv(input_file, chunksize=chunk_size)

for i, chunk in enumerate(reader, 1):
    chunk_file = chunks_dir / f"chunk_{i:02d}.csv"
    chunk.to_csv(chunk_file, index=False)
    print(f"✅ Saved: {chunk_file}")
