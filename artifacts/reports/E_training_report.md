# Speculative Component Training Report

- **Target Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Speculative Architecture:** Medusa Multi-Head (3 heads)
- **Trainable Parameters:** `707,198,976`
- **Precision:** Mixed Precision (`bfloat16`)
- **Optimizer:** AdamW (lr=0.001, weight_decay=1e-4, betas=(0.9, 0.95))
- **Scheduler:** CosineAnnealingLR
- **Gradient Clipping:** `1.0`
- **Epochs:** `10`
- **Wall-Clock Duration:** `95.29` s
- **Best Validation Loss:** `29.4073`

## Validation Accuracy by Head Position

- **Head 0 (Lookahead +1):** `3.95%`
- **Head 1 (Lookahead +2):** `3.58%`
- **Head 2 (Lookahead +3):** `0.74%`

## Training History

| Epoch | Train Loss | Val Loss | Head 0 Val Acc | Head 1 Val Acc | Head 2 Val Acc | LR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 32.2609 | 30.5727 | 1.70% | 1.13% | 1.13% | 9.77e-04 |
| 2 | 7.6415 | 29.4073 | 1.11% | 0.93% | 0.56% | 9.09e-04 |
| 3 | 2.4892 | 32.3846 | 3.57% | 0.93% | 0.37% | 8.04e-04 |
| 4 | 1.0623 | 34.5955 | 3.56% | 2.44% | 0.56% | 6.72e-04 |
| 5 | 0.4988 | 35.4282 | 3.76% | 2.45% | 0.37% | 5.25e-04 |
| 6 | 0.2326 | 35.5132 | 3.95% | 2.07% | 0.37% | 3.78e-04 |
| 7 | 0.1354 | 35.5568 | 3.94% | 3.20% | 0.56% | 2.46e-04 |
| 8 | 0.0616 | 35.6223 | 4.12% | 3.58% | 1.12% | 1.41e-04 |
| 9 | 0.0268 | 35.6579 | 3.95% | 3.58% | 0.74% | 7.32e-05 |
| 10 | 0.0186 | 35.7080 | 3.95% | 3.58% | 0.74% | 5.00e-05 |
