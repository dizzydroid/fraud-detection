# Credit-Card Fraud Detection (Recall-First)

This repo reproduces **Chung & Lee 2023** (Sensors 23-7788) on the PaySim dataset,
then packages the three-model voting strategy into a CLI pipeline.

```bash
make train && make evaluate
```
> artifacts/knn.pkl  lda.pkl  lr.pkl <br />
> results/metrics.json

**Target label** ```isFraud``` (**1 = fraudulent**) <br />
**Primary metric** Recall ≥ 0.93 (**paper’s worst-case**).

See [docs/](docs/) for literature notes and [slides/](slides/) for the final deck.
