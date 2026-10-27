# Data Generation Report

- **Target Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Total Extracted Trajectories:** `28`
- **Total Tokens Extracted:** `2821`
- **Train / Val Split:** `22` train / `6` val
- **Storage Format:** Chunked PyTorch Tensors (`torch.bfloat16`) with SHA-256 Checksums
- **Total Storage Size:** `8.33` MB
- **Extraction Time:** `0.92` s
- **Feature Dimension:** `1536` (Layer `28`)
- **Vocab Dimension:** `151936`

## Chunk Manifest

| Chunk File | Samples | Tokens | Size (MB) | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- |
| `train_chunk_0000.pt` | 10 | 993 | 2.93 | `f7742d1c564d1130...` |
| `train_chunk_0001.pt` | 10 | 1024 | 3.02 | `d36dde1e9bca4eb3...` |
| `train_chunk_0002.pt` | 2 | 202 | 0.60 | `6e40160da51fa8c1...` |
| `val_chunk_0000.pt` | 6 | 602 | 1.78 | `84b3908d489c97f4...` |
