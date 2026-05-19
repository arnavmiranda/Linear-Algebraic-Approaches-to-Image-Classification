# Linear Algebraic Approaches to Image Classification

A deep-dive into the linear algebra underlying convolutional neural networks. This project trains a small 2-layer CNN on MNIST, fully deconstructs its learned weights using **Singular Value Decomposition (SVD)**, and then connects those SVD components back to the original dataset — asking what V, σ, and U actually mean in terms of real digit images.

The longer-term goal this work points toward: **can we derive these weight matrices analytically from the data, without ever running gradient descent?**

---

## Core Insight

Every convolutional weight matrix **W** (shape `out_ch × (in_ch·kH·kW)`) admits an exact decomposition:

```
W  =  U · diag(σ) · Vᵀ
```

This is not an approximation — it is an identity. The forward pass of each conv layer is just three matrix multiplications:

```
1.  z  =  Vᵀ p          project patch p onto input eigenfilter basis
2.  s  =  diag(σ) z     scale each coordinate by its singular value
3.  y  =  U s  =  W p   rotate into output feature space
4.  h  =  ReLU(y)       carve piecewise-linear subspace
```

**Training by gradient descent discovers `{U, σ, V}` — the singular vectors and values — as the natural parameterisation of the filters. This project makes that explicit and asks: what did the network actually find?**

---

## What Each SVD Component Means

| Component | Shape | Meaning |
|-----------|-------|---------|
| **V** (right singular vectors) | `(r, in_ch·kH·kW)` | **Eigenfilters** — orthogonal spatial patterns the layer is sensitive to, ranked by importance. Each row is a direction in patch space. |
| **σ** (singular values) | `(r,)` | **Importance weights** — σᵢ measures how much energy the network places in eigenfilter direction vᵢ. Decay rate reveals effective dimensionality. |
| **U** (left singular vectors) | `(out_ch, r)` | **Feature directions** — orthogonal basis for the output feature space. Determines how filter responses are routed to the next layer. |

---

## What the Network Actually Learned (Phase 9 Results)

After connecting U, V, σ back to 2,000 MNIST test images:

### V — Eigenfilters vs real patches

The `top_patches_eigenfilters.png` figure shows, for each eigenfilter v_i, the MNIST patches that project most strongly onto it (positive and negative). This makes the abstract singular vectors concrete — you can visually read off what spatial pattern each direction is "looking for" in the image.

### σ — Which eigenfilters each digit uses

The per-class eigenfilter response heatmap (`class_eigenfilter_heatmap.png`) shows the mean absolute projection of each digit class onto each eigenfilter direction:

- **v3 dominates for all classes** — it captures a general stroke structure shared by every digit
- **v1 and v5 are most discriminative** — digit 0 and 8 (rounder) have noticeably higher v1 response than digit 1 and 7 (thinner), which aligns with the geometric difference between them
- The σᵢ values weight these responses: directions the network found important get amplified, less useful ones get suppressed

### U — Does the output basis separate digit classes?

The U-space scatter (`uspace_class_scatter.png`) projects every test image's Layer 1 activations onto the left singular vectors. The degree of class separation visible here tells you how much discriminative structure is already present after just one conv layer.

### The Key Finding: Network Eigenfilters ≠ Data PCA

The most important result from Phase 9:

```
Mean alignment between weight eigenfilters and data PCA directions = 0.29
```

If the network simply rediscovered the principal components of the raw MNIST patch distribution, this number would be ~1.0. At 0.29, it is telling us:

> **Gradient descent found directions that maximise classification accuracy, not directions that maximise variance in the data. The two are largely unrelated.**

This is precisely why a CNN outperforms a naive PCA classifier on MNIST. The eigenfilters V are not Fourier modes or PCA components — they are task-specific discriminative directions that the network discovered by optimising the cross-entropy loss.

---

## Project Structure

```
.
├── envision/
│   ├── model.py            # TinyConvNet: 2-layer CNN (1→8→16 channels, 3×3 kernels)
│   ├── train.py            # MNIST training loop + evaluation
│   ├── svd_analysis.py     # SVD decomposition, low-rank approx, im2col, subspace geometry
│   ├── analytical.py       # AnalyticalNet: full forward pass expressed in SVD coordinates
│   ├── dataset_analysis.py # Phase 9: connects U/V/σ back to actual MNIST images
│   └── visualize.py        # All matplotlib figures (11 total)
├── main.py                 # End-to-end pipeline (9 phases)
├── requirements.txt
└── README.md
```

Generated at runtime (gitignored):
```
data/                              # MNIST download
outputs/
├── model.pth                      # Trained weights
├── filters_layer1.png             # Learned conv filters
├── eigenfilters_layer1.png        # Right singular vectors (V rows)
├── singular_spectrum.png          # σᵢ bar chart + cumulative energy
├── lowrank_recon_layer1.png       # Rank-k filter reconstructions
├── svd_identity_layer1/2.png      # W vs U·Σ·Vᵀ visual verification
├── relu_subspace.png              # ReLU activation patterns
├── layer_composition.png          # Spectral energy across both layers
├── top_patches_eigenfilters.png   # MNIST patches that activate each V row
├── class_eigenfilter_heatmap.png  # Per-digit response to each eigenfilter
├── uspace_class_scatter.png       # Digit classes in U-space
└── data_vs_weight_svd.png         # Data PCA vs learned eigenfilters
```

---

## Setup & Running

```bash
pip install -r requirements.txt

# Full run: train 20 epochs + all 9 phases + 11 figures
python main.py

# Skip training (reuse saved weights)
python main.py --skip-train

# Custom epoch count
python main.py --epochs 15
```

MNIST data is downloaded automatically on first run.

---

## The 9-Phase Pipeline

| Phase | What it does |
|-------|-------------|
| **1 — Train** | Train `TinyConvNet` on MNIST; print per-epoch loss and test accuracy |
| **2 — SVD** | Decompose both conv layers; print singular values, cumulative energy, effective rank |
| **3 — Analytical** | Build `AnalyticalNet`; verify `W = U·Σ·Vᵀ` holds to machine precision |
| **4 — Verify** | Confirm analytical `im2col → W·p` matches PyTorch conv output (max diff < 1e-6) |
| **5 — Low-rank** | Sweep rank k = 1…r; print reconstruction error vs cumulative energy |
| **6 — ReLU** | Analyse how ReLU partitions pre-activation space into piecewise-linear regions |
| **7 — Composition** | Verify Layer 2 analytically; confirm Parseval identity `‖W‖²_F = Σ σᵢ²` |
| **8 — Plots** | Generate weight-space figures to `outputs/` |
| **9 — Dataset** | Connect U/V/σ to actual MNIST data: top patches, per-class responses, U-scatter, data PCA alignment |

---

## Key Results (20-epoch run, 98.71% test accuracy)

- **W = U·Σ·Vᵀ to machine epsilon**: max error ~8.9e-08 (L1), ~1.8e-07 (L2)
- **Analytical vs CNN forward pass**: max diff < 1e-6 for both layers
- **Singular value spectra**: Layer 1 `[2.17, 1.75, 1.50, 1.04, 0.79, 0.62, 0.53, 0.42]`; 6 components = 95% energy
- **ReLU creates 40 distinct linear regions** out of a possible 256 across 500 MNIST patches
- **Data PCA alignment = 0.29** — the network's eigenfilters diverge strongly from raw PCA, confirming they encode discrimination not variance

---

## The Road to Deriving Weights Analytically

This project establishes the foundation. The open question it points toward is:

> *Can we construct {U, σ, V} directly from the data — without training — and still classify well?*

The Phase 9 results give a clear picture of what that would require and where the hard problems are:

### What we know from this work

1. **V (eigenfilters) must be discriminative, not just variance-maximising.** Data PCA gives alignment ~0.29 with the trained V — so simply running SVD on the raw patch matrix won't work. The directions that matter for classification are not the directions of maximum variance.

2. **σ (singular values) must reflect class separability, not data energy.** The trained σᵢ weight directions by how much they help distinguish digits. A purely data-driven σ (e.g. from PCA eigenvalues) would weight directions by how much they vary across the whole dataset — which is a different and weaker objective.

3. **U (feature directions) organises the output space for the classifier.** U needs to route filter responses such that the downstream FC layer can linearly separate the 10 classes. This is implicitly learned end-to-end — analytically specifying it requires knowing the class structure in advance.

### Concrete next steps

**Step 1 — Per-class subspace classifier (baseline)**
Build V directly from the data: for each digit class, form a matrix of training patches, take its top-k right singular vectors. Classify a new image by projecting its patches onto each class subspace and picking the closest one. No gradient descent at all. Expected accuracy: ~94–96%, lower than the CNN because V is variance-maximising per class, not jointly discriminative.

**Step 2 — Supervised eigenfilter derivation**
Instead of PCA per class, solve for V that maximises *between-class* variance relative to *within-class* variance — this is Linear Discriminant Analysis (LDA) in patch space. The resulting V rows are analytically derived discriminative eigenfilters. σ can be set to the LDA eigenvalues.

**Step 3 — Analytical U from class geometry**
Given V and σ from Step 2, derive U as the matrix that maps filter responses to a space where the 10 class centroids are maximally separated. This is essentially a Procrustes alignment problem between the filter response space and a target class-label space.

**Step 4 — Close the loop**
Reconstruct W = U·Σ·Vᵀ from the analytically derived components. Run the CNN forward pass using this W (no training). Measure accuracy gap vs the trained model. The gap quantifies exactly what gradient descent adds on top of what linear algebra alone can provide.

### Why this matters

If Steps 1–4 work, the result is a classifier whose weights have a closed-form derivation from the data distribution. That means:
- **Interpretability**: every weight has an explicit geometric meaning derived from class structure
- **No training instability**: no learning rate, no random init, no convergence questions
- **Theoretical guarantees**: accuracy bounds derivable from the data's class geometry

The missing piece is the non-linearity. ReLU allows the network to combine these linear directions in an input-dependent way, which is what pushes accuracy from ~96% to ~99%. Understanding exactly how ReLU interacts with the analytically derived {U, σ, V} — and whether a non-linear counterpart can also be analytically designed — is the open research question this project is building toward.

---

## Background

Convolutional filters act as linear operators on local image patches via `im2col`. The SVD of that operator is not just an analysis tool — it is the parameterisation gradient descent implicitly finds. Related work:

- **Eigenfaces (Turk & Pentland 1991)** — PCA-based face recognition; analytically derived V from data covariance, fixed σ, no training
- **Scattering networks (Mallat 2012)** — analytically designed filter banks using wavelets; provably stable, no training
- **NTK (Jacot et al. 2018)** — in the infinite-width limit, trained network behaviour has a closed-form description
- **LDA-based filter design** — supervised derivation of discriminative filters from class statistics
