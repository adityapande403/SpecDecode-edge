# Verification Correctness & Semantic Preservation

## 1. Equivalence Contract

In speculative decoding, the target language model must remain authoritative.
For greedy decoding ($T = 0$):
Let the target model's true conditional distribution be $P_{target}(y \mid x_{<t})$.
The speculative draft heads propose candidate sequence $(c_1, \dots, c_K)$.

The verifier executes the following contract:
```python
for i in range(K):
    authoritative_token = argmax(P_target(· | x_prefix, c_1, ..., c_{i-1}))
    if c_i == authoritative_token:
        accept(c_i)
    else:
        accept(authoritative_token)  # Emit target model's true greedy token
        reject(c_{i+1}, ..., c_K)
        rollback_kv_cache()
        break
```

### Mathematical Guarantee
Every token appended to the generated sequence is either:
1. A candidate token strictly validated to match $\operatorname{argmax} P_{target}$, OR
2. The direct $\operatorname{argmax}$ token emitted by $P_{target}$ upon candidate rejection.

Under infinite-precision arithmetic, the output sequence of speculative decoding is mathematically identical to standard greedy autoregressive decoding.
