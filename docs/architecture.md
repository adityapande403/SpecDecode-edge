# System Architecture: SpecDecode Edge

## 1. High-Level Architecture Overview

SpecDecode Edge is an end-to-end framework for speculative decoding of LLMs on resource-constrained edge hardware.

```
+-------------------------------------------------------------------------------+
|                            User Prompt Prefix                                 |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                          Target Model Prefill                                 |
|               Authoritative Target LLM (Qwen2.5-1.5B Frozen)                  |
|                         Generates: Token y_0, Hidden h_0                       |
+---------------------------------------+---------------------------------------+
                                        |
                   +--------------------+--------------------+
                   |                                         |
                   v                                         v
+-------------------------------------+   +-------------------------------------+
|        Medusa Multi-Head            |   |           Candidate Proposals       |
|  Head 0: predicts offset t+2        |   |    c_1 = argmax(Head_0(h_0))        |
|  Head 1: predicts offset t+3        |-->|    c_2 = argmax(Head_1(h_0))        |
|  Head 2: predicts offset t+4        |   |    c_3 = argmax(Head_2(h_0))        |
+-------------------------------------+   +------------------+------------------+
                                                             |
                                                             v
+-------------------------------------------------------------------------------+
|                        Speculative Verification Step                          |
|             Target Model processes: [y_0, c_1, c_2, c_3] in single pass       |
|                                                                               |
|   Pos 0: Target logit L_0 -> y_1* == c_1 ?                                    |
|          Match: Accept c_1, proceed to Pos 1.                                 |
|          Mismatch: Reject c_1..c_3, emit authoritative y_1*, rollback cache.  |
|                                                                               |
|   Pos 1: Target logit L_1 -> y_2* == c_2 ?                                    |
|          Match: Accept c_2, proceed to Pos 2.                                 |
|          Mismatch: Reject c_2..c_3, emit authoritative y_2*, rollback cache.  |
|                                                                               |
|   Pos 2: Target logit L_2 -> y_3* == c_3 ?                                    |
|          Match: Accept c_3, accept bonus token from L_3.                       |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                      KV Cache Dynamic Management                              |
|           `crop_cache`: Truncates past_key_values to accepted boundary         |
|              Strict Guarantee: Zero accumulation of rejected states           |
+-------------------------------------------------------------------------------+
```

## 2. Key Modules

- `models/medusa_head.py`: ResBlock MLP modules with SiLU activation and projection to vocab.
- `models/medusa_model.py`: Composite module uniting the frozen base LLM and trainable Medusa heads.
- `trajectory/collector.py`: Streaming hidden state and token trajectory extraction pipeline.
- `training/trainer.py`: Multi-head trainer with AdamW, Cosine Annealing, and validation early stopping.
- `losses/medusa_loss.py`: Truncated multi-step cross-entropy with mask support and exponential decay weights.
- `verification/verifier.py`: Authoritative greedy and stochastic verification algorithms.
- `inference/speculative_generator.py`: Speculative decoding generation loop with KV cache pruning.
- `inference/baseline_generator.py`: Standard autoregressive generation engine with TTFT and TPS profiling.
