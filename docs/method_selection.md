# Speculative Decoding Method Selection: Medusa Multi-Head Architecture

## Overview
This document specifies the chosen speculative-decoding architecture for **SpecDecode Edge**: **Medusa-style Multi-Head Speculative Decoding**.

## 1. Mathematical Formulation

Let the target language model parameters be $\Theta_{target}$ (frozen during draft training).
At generation step $t$, given the token prefix $x_{1:t}$, the target model forward pass computes:
$$h_t = \mathrm{Transformer}(x_{1:t}; \Theta_{target})[-1]$$
$$p_t^{(0)} = \operatorname{softmax}(\mathrm{LM\_Head}(h_t))$$

The standard next token is:
$$x_{t+1} = \operatorname{argmax} p_t^{(0)}$$

### Medusa Multi-Head Extension
Medusa introduces $K$ auxiliary prediction heads: $\Phi = \{\Phi_1, \dots, \Phi_K\}$.
Head $k \in \{1, \dots, K\}$ predicts the token at offset $t + k + 1$:
$$z_t^{(k)} = h_t + \operatorname{SiLU}(W_1^{(k)} h_t + b_1^{(k)})$$
$$p_t^{(k)} = \operatorname{softmax}(W_2^{(k)} z_t^{(k)} + b_2^{(k)})$$

Thus, in a single forward evaluation at position $t$, we obtain candidate proposals:
$$c_1 = \operatorname{argmax} p_t^{(1)}$$
$$c_2 = \operatorname{argmax} p_t^{(2)}$$
$$\dots$$
$$c_K = \operatorname{argmax} p_t^{(K)}$$

### Speculative Candidate Verification
The candidate sequence $(c_1, \dots, c_K)$ is concatenated to prefix $x_{1:t+1}$ and evaluated by the authoritative target model in a **single forward step** (using tree attention or sequential prefix extension with KV caching).

For each candidate position $j \in \{1, \dots, K\}$:
The target model computes true probability distribution $q_j = P_{target}(\cdot \mid x_{1:t+1}, c_1, \dots, c_{j-1})$.
1. If $c_j == \operatorname{argmax}(q_j)$, candidate $c_j$ is **accepted**.
2. If $c_j \neq \operatorname{argmax}(q_j)$, candidate $c_j$ is **rejected**. The authoritative target model emits $x^* = \operatorname{argmax}(q_j)$ as the true next token, all subsequent candidates $c_{j+1}, \dots, c_K$ are discarded, and the KV cache is rolled back to position $t + j + 1$.

### Invariance Guarantee
Because candidate tokens are accepted **if and only if** they match the target model's authoritative greedy choice, the resulting token sequence is mathematically identical to standard greedy autoregressive decoding.
