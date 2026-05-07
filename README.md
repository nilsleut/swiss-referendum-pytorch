# Swiss Referendum PyTorch

Predicting Swiss referendum vote-share percentages from socioeconomic indicators — a Matura thesis project in two stages.

**Stage 1** (`temp/KNN.py`): a multi-layer perceptron built from scratch in NumPy, including manual backpropagation, six optimizers, and conjugate-gradient line search.

**Stage 2** (`feature/`): a clean PyTorch reimplementation that uses autograd, systematic hyperparameter search, and mutual-information feature selection to improve predictive performance.

---

## Task

Given a Swiss referendum, predict the final yes-vote share (%) from 584 features covering:

- Socioeconomic indicators (GDP, inflation, employment)
- Demographic data (population, age distribution)
- National Council seat distribution by party
- Historical voting patterns

**Dataset:** 6,083 referenda · 584 features · target range 5 – 90 %

---

## Repository structure

```
swiss-referendum-pytorch/
├── data/                            # CSV data (gitignored)
│   ├── grosserDatensatzEingabe.csv  # 6 083 × 584 feature matrix
│   └── grosserDatensatzAusgabe.csv  # vote-share targets
│
├── feature/                         # PyTorch implementation
│   ├── dataloader/
│   │   └── dataset.py               # SwissReferendumDataset + build_dataloaders()
│   ├── model/
│   │   └── mlp.py                   # MLP (BatchNorm, dropout, optional ResNet blocks)
│   └── experiments/
│       └── train.py                 # Training loop, ensemble, CLI
│
└── temp/                            # From-scratch NumPy implementation
    ├── KNN.py                       # MLP class (backprop, 6 optimizers, CG line search)
    ├── KNNexe.py                    # Example training script
    ├── Kapitel 2/                   # Educational visualizations
    └── Kapitel 5/                   # Hyperparameter grid-search experiments
```

---

## Results

| Configuration | R² | RMSE |
|---|---|---|
| NumPy baseline (KNN.py, 4×50, Adam) | 0.296 | 10.8 % |
| PyTorch: + BatchNorm + wider arch (256-128-64-32) + dropout | 0.343 | 10.4 % |
| + Feature selection: top-100 by mutual information | 0.362 | 10.3 % |
| + CosineAnnealingLR (smoother than ReduceLROnPlateau) | 0.362 | 10.3 % |
| **Ensemble of 5 models** | **0.368** | **10.2 %** |

Key findings from the hyperparameter search:

- **Feature selection is the single biggest lever** — reducing 584 → 100 features by mutual information with the target adds ~2 R² points. The original feature matrix is 96 % sparse (one-hot encoded categories), and most columns are noise for the MLP.
- **ReduceLROnPlateau with patience=10 kills training** — it halves the learning rate by epoch 20, before the model finds its best minimum. CosineAnnealingLR decays smoothly without hard cutoffs.
- **Residual connections hurt here** — the sparse binary inputs benefit from learning sparse internal representations; skip connections fight against this.
- **Input BatchNorm hurts** — the input is already min-max normalized to [0, 1]; adding BN on top causes a loss spike at epoch 1 and degrades final performance.

---

## Setup

```bash
pip install torch numpy matplotlib
```

No GPU required — all experiments ran on CPU in under 3 minutes per run.

---

## Usage

**Single model (best config, ~90 s on CPU):**

```bash
python -m feature.experiments.train
```

**Ensemble of 5 models (~7 min, R²=0.368):**

```bash
python -m feature.experiments.train --ensemble --n_seeds 5
```

**Custom configuration:**

```bash
python -m feature.experiments.train \
  --hidden 256 128 64 32 \
  --dropout 0.2 \
  --norm batch \
  --lr 3e-4 \
  --n_features 100 \
  --scheduler cosine \
  --max_epochs 300 \
  --patience 40
```

**Key CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--hidden` | `256 128 64 32` | Hidden layer sizes |
| `--dropout` | `0.2` | Dropout probability |
| `--norm` | `batch` | Normalization: `none`, `batch`, `layer` |
| `--lr` | `3e-4` | Learning rate |
| `--n_features` | `100` | Top-k features by MI; `0` = all 584 |
| `--scheduler` | `cosine` | LR schedule: `cosine`, `plateau`, `none` |
| `--optimizer` | `adam` | `adam`, `rmsprop`, `sgd`, `adagrad` |
| `--loss` | `mse` | `mse`, `huber`, `mae` |
| `--ensemble` | off | Average N independently trained models |
| `--n_seeds` | `5` | Number of models in the ensemble |

---

## Comparison: NumPy vs PyTorch implementation

| Feature | NumPy (KNN.py) | PyTorch (feature/) |
|---|---|---|
| Backpropagation | Manual chain rule | `loss.backward()` |
| GPU support | No | CUDA / MPS auto-detected |
| Optimizers | SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam, CG | PyTorch builtins |
| LR scheduling | Manual linear decay | CosineAnnealingLR |
| Gradient clipping | CG path only | Every step |
| Normalization | None | BatchNorm / LayerNorm (optional) |
| Weight init | He / Xavier | Same, via `kaiming_normal_` / `xavier_normal_` |
| Feature selection | None | Top-k by mutual information |
| Checkpointing | In-memory (best weights) | `best_model.pt` saved to disk |
| Ensemble | No | `ensemble_train()` |
