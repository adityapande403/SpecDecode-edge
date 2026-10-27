# Trajectory Data Generation Pipeline

## 1. Overview

The trajectory data generation pipeline extracts frozen target model representations and token alignment labels for training speculative draft components.

## 2. Pipeline Stages

1. **Seed Prompt Expansion:**
   Diverse seed prompts covering QA, code generation, mathematical reasoning, technical explanations, and structured schemas.
2. **Authoritative Trajectory Completion:**
   The target model runs autoregressive generation with greedy decoding to generate natural continuous language trajectories.
3. **Hidden State Extraction:**
   The full trajectory is passed through the target model forward pass. The final layer hidden states $h_t \in \mathbb{R}^{d}$ before the LM head are extracted.
4. **Chunking and Storage:**
   Features and token labels are partitioned into chunk files (`train_chunk_*.pt`, `val_chunk_*.pt`) in `data/trajectories/`.
5. **Provenance and Manifest:**
   Every chunk's sample count, token count, file size, and SHA-256 hash are recorded in `data/trajectories/manifest.json`.
