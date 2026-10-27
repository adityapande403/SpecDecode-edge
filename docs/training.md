# Training Methodology and Loss Function Formulation

## 1. Loss Function Formulation

In Medusa multi-head speculative decoding, $K$ prediction heads are trained to predict lookahead tokens while the base target model is kept frozen.

Given input token sequence $X = (x_1, \dots, x_N)$ and extracted final hidden states $H = (h_1, \dots, h_N)$:

For each head $k \in \{0, \dots, K-1\}$:
The head predicts token $x_{t + k + 2}$ given hidden state $h_t$.

The prediction logits are:
$$\hat{Y}^{(k)} = \operatorname{Head}_k(h_t)$$

The cross-entropy loss for head $k$ is:
$$\mathcal{L}_{CE}^{(k)} = - \frac{1}{\sum_{t} M_t} \sum_{t} M_t \log P_k(x_{t + k + 2} \mid h_t)$$

where $M_t \in \{0, 1\}$ is a binary mask that is 0 for prompt/padding tokens and 1 for generated target tokens.

### Multi-Head Weighted Objective
The composite training loss is:
$$\mathcal{L}_{total} = \sum_{k=0}^{K-1} \lambda_k \mathcal{L}_{CE}^{(k)}$$

with geometric decay weighting:
$$\lambda_k = \gamma^k, \quad \gamma = 0.8$$

This prioritizes accuracy on near-term speculative positions (which have the highest statistical probability of being evaluated and accepted) while still training longer lookaheads.

## 2. Optimization Configuration
- **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.95, \epsilon = 10^{-8}$)
- **Learning Rate:** $1.0 \times 10^{-3}$ with Cosine Annealing decay down to $5.0 \times 10^{-5}$
- **Weight Decay:** $1.0 \times 10^{-4}$
- **Gradient Clipping:** Maximum norm $1.0$
- **Mixed Precision:** `torch.bfloat16` via `torch.amp.autocast`
