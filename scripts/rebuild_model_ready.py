import pandas as pd
from pathlib import Path

chunks_dir = Path("data/chunks")
output_file = Path("data/model_ready.csv")

print("🧩 Rebuilding model_ready.csv from chunks...")

chunk_files = sorted(chunks_dir.glob("chunk_*.csv"))
print(f"📦 Found {len(chunk_files)} chunks")

dfs = []
for i, file in enumerate(chunk_files, 1):
    print(f"📥 Reading chunk {i}: {file.name}")
    df_chunk = pd.read_csv(file)
    dfs.append(df_chunk)

df_final = pd.concat(dfs, ignore_index=True)
df_final.to_csv(output_file, index=False)

print(f"✅ All chunks merged. Final shape: {df_final.shape}")
print(f"📁 Saved to {output_file}")
