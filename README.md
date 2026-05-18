# Linear Algebraic Approaches to Image Classification

A deep-dive into the linear algebra underlying convolutional neural networks. This project trains a small 2-layer CNN on MNIST and then fully deconstructs its learned weights using **Singular Value Decomposition (SVD)**, demonstrating that every convolutional layer *is* an SVD — nothing more, nothing less.

---

## Core Insight

Every convolutional weight matrix **W** (shape `out_ch × (in_ch·kH·kW)`) admits an exact decomposition:

```
W  =  U · diag(σ) · Vᵀ
```

This is not an approximation — it is an identity. The forward pass of each conv layer can be written entirely in terms of `{U, σ, V}`:

```
1.  z  =  Vᵀ p          project patch p onto input eigenfilter basis
2.  s  =  diag(σ) z     scale each coordinate by its singular value
3.  y  =  U s  =  W p   rotate into output feature space
4.  h  =  ReLU(y)       carve piecewise-linear subspace
```

**Training by gradient descent discovers `{U, σ, V}` — the singular vectors and values — as its natural parameterisation of the filters.**

---

## What Each SVD Component Means

| Component | Shape | Meaning |
|-----------|-------|---------|
| **V** (right singular vectors) | `(r, in_ch·kH·kW)` | **Eigenfilters** — orthogonal spatial patterns the layer is sensitive to, ranked by importance |
| **σ** (singular values) | `(r,)` | **Importance weights** — σᵢ measures how much energy the network places in eigenfilter direction vᵢ |
| **U** (left singular vectors) | `(out_ch, r)` | **Feature directions** — orthogonal basis for the output feature space |

The decay rate of σᵢ reveals the **effective dimensionality** of the layer: how many directions actually matter.

---

## Project Structure

```
.
├── envision/               # Core Python package
│   ├── model.py            # TinyConvNet: 2-layer CNN (1→8→16 channels, 3×3 kernels)
│   ├── train.py            # MNIST training loop + evaluation
│   ├── svd_analysis.py     # SVD decomposition, low-rank approx, im2col, subspace geometry
│   ├── analytical.py       # AnalyticalNet: full forward pass in SVD coordinates
│   └── visualize.py        # All matplotlib figures
├── main.py                 # End-to-end pipeline (8 phases)
├── requirements.txt
└── README.md
```

Generated at runtime (gitignored):
```
data/                       # MNIST download
outputs/
├── model.pth               # Trained weights
├── filters_layer1.png      # Learned conv filters
├── eigenfilters_layer1.png # Right singular vectors (V rows)
├── singular_spectrum.png   # σᵢ bar chart + cumulative energy
├── lowrank_recon_layer1.png# Rank-k filter reconstructions
├── svd_identity_layer1.png # W vs U·Σ·Vᵀ visual verification
├── svd_identity_layer2.png
├── relu_subspace.png       # ReLU activation patterns & distributions
└── layer_composition.png   # Spectral energy across both layers
```

---

## Setup

```bash
pip install -r requirements.txt
```

Requirements: `torch`, `torchvision`, `numpy`, `matplotlib`, `scipy`.

---

## Running

```bash
# Full run: train 20 epochs + full analysis + all figures
python main.py

# Skip training (reuse saved weights)
python main.py --skip-train

# Custom epoch count
python main.py --epochs 15
```

MNIST data is downloaded automatically on the first run.

---

## The 8-Phase Pipeline

| Phase | What it does |
|-------|-------------|
| **1 — Train** | Train `TinyConvNet` on MNIST for N epochs; print per-epoch loss and test accuracy |
| **2 — SVD** | Decompose both conv layers; print singular values, cumulative energy, effective rank |
| **3 — Analytical** | Build `AnalyticalNet`; verify `W = U·Σ·Vᵀ` holds to machine precision; print two-layer composition in SVD coordinates |
| **4 — Verify** | Confirm analytical `im2col → W·p` output matches PyTorch conv output numerically (max diff < 1e-4) |
| **5 — Low-rank** | Sweep rank k = 1…r; print relative reconstruction error and cumulative energy captured |
| **6 — ReLU** | Analyse how ReLU partitions the pre-activation space into piecewise-linear regions; count unique activation masks |
| **7 — Composition** | Verify Layer 2 analytically; print Parseval identity `‖W‖²_F = Σ σᵢ²` for both layers |
| **8 — Plots** | Generate all figures to `outputs/` |

---

## The Network

```
TinyConvNet
  conv1: Conv2d(1,  8,  3×3, bias=False)   →  (B, 8,  26, 26)
  ReLU + MaxPool(2)                         →  (B, 8,  13, 13)
  conv2: Conv2d(8,  16, 3×3, bias=False)   →  (B, 16, 11, 11)
  ReLU + MaxPool(2)                         →  (B, 16,  5,  5)
  fc:   Linear(400, 10)
```

Bias is disabled in both conv layers so the weight matrix alone captures the full transformation — no affine shift to account for.

| Layer | Weight shape | SVD shapes |
|-------|-------------|------------|
| conv1 | (8, 9) | U:(8,8), σ:(8,), Vᵀ:(8,9) |
| conv2 | (16, 72) | U:(16,16), σ:(16,), Vᵀ:(16,72) |

---

## Key Results (20-epoch run, 98.71% test accuracy)

- **W = U·Σ·Vᵀ holds to machine epsilon** for both layers: max error ~8.9e-08 (L1), ~1.8e-07 (L2).
- **Singular value spectra** after training:
  - Layer 1: `[2.170, 1.753, 1.496, 1.041, 0.787, 0.623, 0.534, 0.424]` — 6 components capture 95% energy
  - Layer 2: `[2.895, 2.480, 2.349, 2.070, 1.924, 1.701, 1.559, 1.534, ...]` — 13 components capture 95% energy, 15/16 for 99%
- **ReLU subspace analysis**: Layer 1 (8 output channels) can create up to 2⁸ = 256 distinct linear regions; in practice only **40 unique activation masks** appear across 500 MNIST patches.
- **Frobenius norm identity** `‖W‖²_F = Σ σᵢ²` verified numerically: L1 = 12.5733 ✓, L2 = 43.7439 ✓
- **Analytical vs CNN forward pass**: max absolute diff < 1e-6 for both layers.

---

## Background

Convolutional filters can be viewed as linear operators acting on local image patches via `im2col`. The SVD of that operator is not just an analysis tool — it *is* the parameterisation that gradient descent finds. This project makes that explicit and verifiable:

- **V** learns which spatial frequency patterns matter (edge detectors, blobs, diagonals…)
- **σ** encodes how strongly the network responds to each pattern
- **U** routes those responses into distinguishable output channels
- **ReLU** then selects a piecewise-linear subspace based on which inputs are active

This viewpoint gives a principled explanation for why low-rank compression of conv layers works: the bottom singular components carry negligible energy and removing them barely changes the output.
