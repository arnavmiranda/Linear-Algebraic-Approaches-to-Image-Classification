"""
Envision  -  2-layer CNN <-> SVD linear-algebra bridge
====================================================

Phases:
  1  Train TinyConvNet on MNIST (or load cached weights)
  2  SVD-decompose both conv layers; print singular spectra
  3  Build AnalyticalNet; confirm W_i = U_i Sigma_i V_i^T exactly
  4  Numerical verification: analytical output == CNN output
  5  Low-rank approximation: how well does rank-k filter bank do?
  6  Non-linear subspace analysis: how ReLU carves linear regions
  7  Two-layer composition expressed in SVD coordinates
  8  Generate all visualisation figures -> outputs/

Run:
    python main.py
    python main.py --skip-train          (reuse outputs/model.pth)
    python main.py --epochs 20
"""

import os, argparse
import numpy as np
import torch

from envision.model      import TinyConvNet
from envision.train      import train, get_loaders
from envision.svd_analysis import analyze_layer, batch_im2col, reconstruction_error, low_rank_approx
from envision.analytical import AnalyticalNet
from envision.visualize  import (
    plot_filters, plot_eigenfilters, plot_spectrum,
    plot_lowrank_recon, plot_svd_identity,
    plot_relu_subspace, plot_layer_composition,
)

os.makedirs('outputs', exist_ok=True)

BANNER = lambda title: print(f"\n{'#'*62}\n#  {title}\n{'#'*62}")


# ============================================================================
#  CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--skip-train', action='store_true',
                   help='Load model.pth instead of re-training')
    p.add_argument('--epochs',  type=int, default=20)
    return p.parse_args()


# ============================================================================
#  PHASE 1  -  Train / Load
# ============================================================================

def phase1_train(skip_train, epochs):
    BANNER("PHASE 1   -   2-Layer CNN on MNIST")

    model = TinyConvNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{model}")
    print(f"\nTotal learnable parameters: {total_params:,}")
    print(f"  conv1 weights : {model.conv1.weight.numel():,}   (8 filters x 1x3x3 = 72 scalars)")
    print(f"  conv2 weights : {model.conv2.weight.numel():,}   (16 filters x 8x3x3 = 1152 scalars)")
    print(f"  fc weights    : {model.fc.weight.numel():,}")

    model_path = 'outputs/model.pth'
    if skip_train and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"\nLoaded weights from {model_path}")
        _, test_loader = get_loaders()
    else:
        model, _, test_loader = train(model, epochs=epochs, save_path=model_path)

    model.eval()
    return model, test_loader


# ============================================================================
#  PHASE 2  -  SVD decomposition
# ============================================================================

def phase2_svd(model):
    BANNER("PHASE 2   -   SVD Decomposition of Convolutional Weights")
    W1, U1, s1, Vt1 = analyze_layer(model.conv1, "Layer 1 (1->8 ch, 3x3 kernel)")
    W2, U2, s2, Vt2 = analyze_layer(model.conv2, "Layer 2 (8->16 ch, 3x3 kernel)")
    return (W1, U1, s1, Vt1), (W2, U2, s2, Vt2)


# ============================================================================
#  PHASE 3  -  Analytical network
# ============================================================================

def phase3_analytical(model):
    BANNER("PHASE 3   -   Analytical Reconstruction Using SVD Constants")
    anet = AnalyticalNet(model)
    anet.report_equivalence()
    anet.report_subspaces()
    anet.two_layer_composition()
    return anet


# ============================================================================
#  PHASE 4  -  Numerical verification
# ============================================================================

def phase4_verify(model, test_loader, W1, U1, s1, Vt1):
    BANNER("PHASE 4   -   Numerical Verification: CNN == Analytical")

    # Grab one batch
    images, labels = next(iter(test_loader))
    images = images[:32]
    imgs_np = images.numpy()[:, 0]   # (32, 28, 28)

    # CNN conv1 output (pre-ReLU)
    with torch.no_grad():
        acts = model.get_activations(images)
    cnn_pre_relu1 = acts['pre_relu1'].numpy()   # (32, 8, 26, 26)

    # Analytical: im2col -> W1 matmul
    patches = batch_im2col(imgs_np, kH=3, kW=3)           # (32*26*26, 9)
    analytical = patches @ W1.T                             # (32*26*26, 8)
    analytical_4d = analytical.reshape(32, 26, 26, 8).transpose(0, 3, 1, 2)

    diff = np.abs(analytical_4d - cnn_pre_relu1)

    print(f"\n  CNN conv1 output shape       : {cnn_pre_relu1.shape}")
    print(f"  Analytical output shape      : {analytical_4d.shape}")
    print(f"  Max absolute difference      : {diff.max():.3e}")
    print(f"  Mean absolute difference     : {diff.mean():.3e}")
    ok = diff.max() < 1e-4
    print(f"  Verdict: {'OK  MATCH  (CNN == W.patch)' if ok else 'FAIL  MISMATCH'}")

    # Extra: confirm W == U Sigma Vt analytically
    W1_from_svd   = (U1 * s1) @ Vt1
    svd_out        = patches @ W1_from_svd.T
    svd_diff       = np.abs(analytical - svd_out)
    print(f"\n  W  vs  U.Sigma.V^T:")
    print(f"    Max |W.p - (USigmaV^T).p| = {svd_diff.max():.3e}")
    print(f"    Verdict: {'OK  W = U.Sigma.V^T CONFIRMED' if svd_diff.max() < 1e-5 else 'FAIL'}")

    return patches, images, acts


# ============================================================================
#  PHASE 5  -  Low-rank approximation
# ============================================================================

def phase5_lowrank(patches, W1, U1, s1, Vt1, W2, U2, s2, Vt2):
    BANNER("PHASE 5   -   Low-Rank Approximation Accuracy")

    sample = patches[:200]

    print(f"\n  Layer 1  (W shape {W1.shape}):")
    print(f"  {'k':>4}  {'energy%':>9}  {'rel_err':>10}")
    print(f"  {'-'*27}")
    for k in range(1, len(s1) + 1):
        W_k  = low_rank_approx(U1, s1, Vt1, k)
        err  = reconstruction_error(W1, W_k, sample)
        ecum = (np.cumsum(s1**2) / (s1**2).sum())[k-1]
        marker = " <- full rank" if k == len(s1) else ""
        print(f"  {k:>4}  {ecum*100:>8.1f}%  {err:>10.6f}{marker}")

    print(f"\n  Layer 2  (W shape {W2.shape})   -  showing top 8:")
    print(f"  {'k':>4}  {'energy%':>9}  {'rel_err':>10}")
    print(f"  {'-'*27}")

    # For layer 2, get patches from conv1 pooled output
    # We'll just generate random patches of the right size for the error analysis
    rng    = np.random.default_rng(42)
    p2_rnd = rng.standard_normal((200, W2.shape[1]))

    for k in [1, 2, 4, 8, 12, 16]:
        if k > len(s2):
            break
        W_k  = low_rank_approx(U2, s2, Vt2, k)
        err  = reconstruction_error(W2, W_k, p2_rnd)
        ecum = (np.cumsum(s2**2) / (s2**2).sum())[k-1]
        marker = " <- full rank" if k == len(s2) else ""
        print(f"  {k:>4}  {ecum*100:>8.1f}%  {err:>10.6f}{marker}")


# ============================================================================
#  PHASE 6  -  Non-linear subspace analysis
# ============================================================================

def phase6_relu(anet, patches):
    BANNER("PHASE 6   -   Non-linear Subspace Analysis (ReLU)")
    pre, post, coords_pre, coords_post = anet.nonlinear_subspace_analysis(patches[:500])
    return pre, post, coords_pre, coords_post


# ============================================================================
#  PHASE 7  -  Two-layer composition (already printed in phase 3)
# ============================================================================

def phase7_composition(anet, patches, acts):
    BANNER("PHASE 7   -   Full Forward Pass in SVD Coordinates")

    imgs  = acts['pool1'].numpy()   # (32, 8, 13, 13)
    N, C, H, W = imgs.shape
    patches2 = batch_im2col(imgs.reshape(N * C, H, W), kH=1, kW=1)  # naive for display
    # proper: extract 3x3 patches across the spatial+channel dims
    out_H2, out_W2 = H - 3 + 1, W - 3 + 1   # = 11
    p2_list = []
    for i in range(N):
        for h in range(out_H2):
            for w in range(out_W2):
                p2_list.append(imgs[i, :, h:h+3, w:w+3].ravel())
    patches2 = np.array(p2_list)   # (N*11*11, 72)

    # Verify layer 2 analytically
    cnn_pre2   = acts['pre_relu2'].numpy()   # (32, 16, 11, 11)
    analytical2 = (patches2 @ anet.L2.W.T).reshape(N, 11, 11, 16).transpose(0, 3, 1, 2)
    diff2 = np.abs(analytical2 - cnn_pre2)

    print(f"\n  Layer 2 analytical verification:")
    print(f"    Max |CNN - analytical| = {diff2.max():.3e}")
    ok = diff2.max() < 1e-4
    print(f"    Verdict: {'OK  MATCH' if ok else 'FAIL  MISMATCH'}")

    print(f"\n  Subspace chain (energy flow):")
    print(f"    L1 Frobenius norm ||W1||_F = {np.linalg.norm(anet.L1.W):.4f}")
    print(f"    L2 Frobenius norm ||W2||_F = {np.linalg.norm(anet.L2.W):.4f}")
    print(f"    Note: ||W||_F^2 = Sigma sigma_i^2 (Parseval-like identity from SVD)")
    print(f"    Verification L1: {np.linalg.norm(anet.L1.W)**2:.4f} vs Sigmasigma^2={np.sum(anet.L1.s**2):.4f}")
    print(f"    Verification L2: {np.linalg.norm(anet.L2.W)**2:.4f} vs Sigmasigma^2={np.sum(anet.L2.s**2):.4f}")

    return patches2


# ============================================================================
#  PHASE 8  -  Visualisations
# ============================================================================

def phase8_plots(W1, U1, s1, Vt1, W2, U2, s2, Vt2,
                 pre, post, coords_pre, coords_post):
    BANNER("PHASE 8   -   Generating Visualisations")
    print()

    plot_filters(W1, "Layer 1: Learned Conv Filters (1x3x3, 8 filters)",
                 "filters_layer1.png")

    plot_eigenfilters(Vt1, s1,
                      "Layer 1: Eigenfilters  V1  (right singular vectors of W1)",
                      "eigenfilters_layer1.png")

    # For layer 2, Vt2 rows have dim 72 = 8x3x3  -  not square, skip image plot
    # Instead show the singular spectrum only
    plot_spectrum(s1, s2, "singular_spectrum.png")

    plot_lowrank_recon(W1, U1, s1, Vt1,
                       "Layer 1: Rank-k Reconstruction of Learned Filters",
                       "lowrank_recon_layer1.png",
                       ks=[1, 2, 4, len(s1)])

    plot_svd_identity(W1, U1, s1, Vt1, "Layer 1", "svd_identity_layer1.png")
    plot_svd_identity(W2, U2, s2, Vt2, "Layer 2", "svd_identity_layer2.png")

    plot_relu_subspace(pre, post, coords_pre, coords_post, "relu_subspace.png")

    plot_layer_composition(s1, s2, "layer_composition.png")


# ============================================================================
#  FINAL SUMMARY
# ============================================================================

def final_summary(s1, s2):
    from envision.svd_analysis import effective_rank
    BANNER("ENVISION   -   CONCLUSION")
    eff1 = effective_rank(s1, 0.99)
    eff2 = effective_rank(s2, 0.99)
    print(f"""
+-------------------------------------------------------------+
|  Core Result                                                |
|                                                             |
|  Every convolutional weight matrix W_i satisfies exactly:  |
|                                                             |
|         W_i  =  U_i . diag(sigma_i) . V_i^T                   |
|                                                             |
|  The SVD constants  {{ U_i, sigma_i, V_i }}  ARE the weights.   |
|  Training discovers them by gradient descent on the loss.   |
|                                                             |
|  Interpretation of each SVD component:                     |
|   V_i   -  right singular vectors = "eigenfilters"           |
|           orthogonal basis for the input patch space.       |
|           Each column v_j is a spatial pattern the layer    |
|           is sensitive to.                                  |
|                                                             |
|   sigma_i   -  singular values = "importance constants"          |
|           sigma_j measures how much energy the network places   |
|           in eigenfilter direction v_j.                     |
|           Decay rate of sigma_i reveals effective dimensionality|
|                                                             |
|   U_i   -  left singular vectors = "feature directions"      |
|           orthogonal basis for the output feature space.    |
|           The network writes its responses in this basis.   |
|                                                             |
|  Non-linearity:                                             |
|   ReLU(W_i p) = ReLU(U_i diag(sigma_i) V_i^T p)               |
|   For each patch p, the active set {{j : sigma_j v_j^T p > 0}} |
|   defines a distinct linear subspace (piecewise-linear map) |
|   => the network carves R^dim into <= 2^out_ch linear regions|
|                                                             |
|  Numbers from this run:                                     |
|   Layer 1  sigma = {np.round(s1, 3)}
|   Layer 2  sigma = {np.round(s2[:8], 3)} ...
|   Effective rank (99% energy): L1={eff1}/{len(s1)}, L2={eff2}/{len(s2)}
+-------------------------------------------------------------+

  Outputs saved to ./outputs/
""")


# ============================================================================
#  ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    args = parse_args()

    model, test_loader                  = phase1_train(args.skip_train, args.epochs)
    (W1, U1, s1, Vt1), (W2, U2, s2, Vt2) = phase2_svd(model)
    anet                                = phase3_analytical(model)
    patches, images, acts               = phase4_verify(model, test_loader, W1, U1, s1, Vt1)
    phase5_lowrank(patches, W1, U1, s1, Vt1, W2, U2, s2, Vt2)
    pre, post, coords_pre, coords_post  = phase6_relu(anet, patches)
    phase7_composition(anet, patches, acts)
    phase8_plots(W1, U1, s1, Vt1, W2, U2, s2, Vt2,
                 pre, post, coords_pre, coords_post)
    final_summary(s1, s2)
