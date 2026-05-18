"""
Analytical reconstruction of the 2-layer CNN using SVD.

The central claim of Envision:

    The convolutional weight matrix W IS the product U Sigma V^T.

    The neural network forward pass can therefore be written entirely in terms
    of the SVD constants {U_i, sigma_i, V_i}:

        Layer i forward:
          1.  z  = V_i^T  p        (project patch p onto input eigenfilter basis)
          2.  s  = diag(sigma_i) z     (scale each coordinate by its singular value)
          3.  y  = U_i    s        (rotate into output feature space)
          4.  h  = ReLU(y)         (piecewise-linear non-linearity)

    For two layers composed:
        h1 = ReLU(U1 Sigma1 V1^T  .  im2col(x))
        h2 = ReLU(U2 Sigma2 V2^T  .  im2col(MaxPool(h1)))
        out = W_fc  h2 + b_fc

    Training optimises W_i = U_i Sigma_i V_i^T jointly.
    The singular values sigma_i are the "natural constants" of each transformation:
    they measure how much energy / discriminative power lives in each
    eigenfilter direction.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .svd_analysis import (
    filter_matrix, svd_decompose, low_rank_approx,
    cumulative_energy, effective_rank, batch_im2col, reconstruction_error,
)


class AnalyticalConvLayer:
    """
    Wraps a trained Conv2d layer and exposes its SVD decomposition.

    Attributes
    ----------
    W     : (out_ch, dim)       original weight matrix
    U     : (out_ch, r)         left  singular vectors
    s     : (r,)                singular values
    Vt    : (r, dim)            right singular vectors (rows = v_i^T)
    k     : int                 number of components used for approximation
    W_k   : (out_ch, dim)       rank-k reconstruction  U_k Sigma_k V_k^T
    """

    def __init__(self, conv_layer, k=None):
        self.W = filter_matrix(conv_layer)
        self.U, self.s, self.Vt = svd_decompose(self.W)
        self.out_ch = self.W.shape[0]
        self.dim    = self.W.shape[1]
        self.k      = k if k is not None else len(self.s)
        self.W_k    = low_rank_approx(self.U, self.s, self.Vt, self.k)

    # -- per-patch forward pass (educational, not batched) ------------------

    def forward_patch_svd(self, patch):
        """
        Explicit 3-step SVD forward pass on one patch.
        Returns z (V-coords), s_scaled (Sigma z), y (U s_scaled).
        Demonstrates: y = U Sigma V^T patch  ==  W patch.
        """
        p  = np.asarray(patch).ravel()
        z  = self.Vt[:self.k] @ p          # (k,)  -  input subspace coordinates
        zs = self.s[:self.k] * z           # (k,)  -  scaled
        y  = self.U[:, :self.k] @ zs       # (out_ch,)  -  output feature space
        return z, zs, y

    # -- batched comparison -------------------------------------------------

    def compare(self, patches):
        """
        patches : (N, dim)
        Returns dict with direct and SVD outputs and error stats.
        """
        direct     = patches @ self.W.T
        analytical = patches @ self.W_k.T
        diff       = direct - analytical
        return {
            'direct':     direct,
            'analytical': analytical,
            'max_err':    np.abs(diff).max(),
            'mean_err':   np.abs(diff).mean(),
            'rel_err':    np.linalg.norm(diff) / (np.linalg.norm(direct) + 1e-12),
        }


class AnalyticalNet:
    """
    Full 2-layer analytical network mirroring TinyConvNet.
    All computation expressed in terms of SVD constants.
    """

    def __init__(self, cnn_model):
        self.L1 = AnalyticalConvLayer(cnn_model.conv1)
        self.L2 = AnalyticalConvLayer(cnn_model.conv2)
        self.fc_W = cnn_model.fc.weight.data.numpy()   # (10, 400)
        self.fc_b = cnn_model.fc.bias.data.numpy()     # (10,)
        self._print_summary()

    def _print_summary(self):
        print("\n" + "=" * 62)
        print("  Analytical Network  -  SVD Parameterisation")
        print("=" * 62)
        print(f"  Layer 1:  W1 = U1 . Sigma1 . V1^T")
        print(f"            U1 {self.L1.U.shape},  sigma1 {self.L1.s.shape},  V1^T {self.L1.Vt.shape}")
        print(f"            sigma1 = {np.round(self.L1.s, 4)}")
        print()
        print(f"  Layer 2:  W2 = U2 . Sigma2 . V2^T")
        print(f"            U2 {self.L2.U.shape},  sigma2 {self.L2.s.shape},  V2^T {self.L2.Vt.shape}")
        print(f"            sigma2 = {np.round(self.L2.s, 4)}")
        print()
        print(f"  FC layer: W_fc {self.fc_W.shape}")

    # -- Phase-3 reports ---------------------------------------------------

    def report_equivalence(self):
        """Show W = U Sigma V^T exactly (machine precision)."""
        print("\n" + "=" * 62)
        print("  Key Result: W_i  ==  U_i . Sigma_i . V_i^T")
        print("=" * 62)
        for layer, name in [(self.L1, "Layer 1"), (self.L2, "Layer 2")]:
            W_rec = (layer.U * layer.s) @ layer.Vt
            err   = np.abs(layer.W - W_rec).max()
            label = "EXACT (machine epsilon)" if err < 1e-5 else f"error = {err:.2e}"
            print(f"  {name}: max|W - USigmaVt| = {err:.2e}   [{label}]")
        print()
        print("  Interpretation:")
        print("    The WEIGHTS themselves ARE the SVD constants:")
        print("    training discovers {U_i, sigma_i, V_i} by optimising W_i.")
        print("    sigma_i encodes how much each eigenfilter direction matters.")

    def report_subspaces(self):
        """Effective rank, energy distribution, subspace interpretation."""
        print("\n" + "=" * 62)
        print("  Subspace Structure")
        print("=" * 62)
        for layer, name in [(self.L1, "Layer 1"), (self.L2, "Layer 2")]:
            eff95 = effective_rank(layer.s, 0.95)
            eff99 = effective_rank(layer.s, 0.99)
            r     = len(layer.s)
            print(f"\n  {name}:")
            print(f"    Input  space: R^{layer.dim}  (in_ch x kH x kW patch)")
            print(f"    Output space: R^{layer.out_ch}  (filter responses)")
            print(f"    Effective rank  95% energy : {eff95}/{r}")
            print(f"    Effective rank  99% energy : {eff99}/{r}")
            print(f"    -> The layer operates in a ~{eff99}-dimensional subspace")
            print(f"      even though the ambient dim is {layer.dim}.")

    def nonlinear_subspace_analysis(self, patches):
        """
        Analyse how ReLU partitions the output space into linear regions.

        Each unique binary activation pattern M in {0,1}^8 defines a
        different effective linear map:  h = diag(M) W patch.
        This is the piecewise-linear structure induced by the nonlinearity.
        """
        print("\n" + "=" * 62)
        print("  Non-linear Subspace Analysis (ReLU)")
        print("=" * 62)

        pre  = patches @ self.L1.W.T    # (N, 8)
        post = np.maximum(0.0, pre)     # (N, 8)

        act_rate     = (pre > 0).mean(axis=0)
        avg_active   = (pre > 0).sum(axis=1).mean()

        # Unique activation patterns -> distinct linear regions
        masks        = (pre > 0).astype(np.uint8)
        unique_masks = np.unique(masks, axis=0)

        # Coordinates in output (left singular vector) basis
        U1          = self.L1.U
        coords_pre  = pre  @ U1    # (N, out_ch)
        coords_post = post @ U1    # (N, out_ch)

        print(f"\n  Patches analysed        : {len(patches)}")
        print(f"  Pre-ReLU  mean / std    : {pre.mean():.4f} / {pre.std():.4f}")
        print(f"  Avg active neurons/patch: {avg_active:.2f} / {self.L1.out_ch}")
        print(f"  Per-filter activation % : {np.round(act_rate*100, 1)}")
        print(f"  Unique activation masks : {len(unique_masks)}  "
              f"(= distinct linear subregions encountered)")
        print()
        print("  Piecewise-linear decomposition:")
        print("    For each patch p, let M(p) = diag(1[W p > 0])  (binary mask)")
        print("    Then:  h(p) = ReLU(W p) = M(p) W p")
        print("    Each mask defines a different effective weight matrix M(p)W,")
        print("    all sharing the same SVD basis {U, V} but with masked sigma values.")

        return pre, post, coords_pre, coords_post

    def two_layer_composition(self):
        """Print the full 2-layer composition in SVD notation."""
        print("\n" + "=" * 62)
        print("  Two-Layer Composition in SVD Coordinates")
        print("=" * 62)
        L1, L2 = self.L1, self.L2
        print(f"""
  Given image x in R^(1x28x28):

  +- Layer 1 -------------------------------------------------+
  |  patches p in R^{L1.dim} (all 3x3 sliding windows of x)        |
  |                                                           |
  |  z1 = V1^T p          project -> R^{L1.Vt.shape[0]:2d}  (input eigenfilter coords) |
  |  s1 = Sigma1 z1          scale   -> R^{len(L1.s):2d}  (singular value weights) |
  |  a1 = U1 s1 = W1 p   lift    -> R^{L1.out_ch:2d}  (filter responses)        |
  |  h1 = ReLU(a1)       carve piecewise-linear subspace     |
  |  -> MaxPool(h1): spatially downsample                      |
  +-----------------------------------------------------------+
           v pooled feature map patches f in R^{L2.dim}
  +- Layer 2 -------------------------------------------------+
  |  z2 = V2^T f                                              |
  |  s2 = Sigma2 z2                                              |
  |  a2 = U2 s2 = W2 f                                       |
  |  h2 = ReLU(a2)                                           |
  |  -> MaxPool(h2) + flatten -> R^400                         |
  +-----------------------------------------------------------+
           v
  output = W_fc . h2 + b_fc  in R^10  (logits)

  The network constants are:
    {{ U1, sigma1, V1 }}  ==  W1   (Layer 1, shape {L1.W.shape})
    {{ U2, sigma2, V2 }}  ==  W2   (Layer 2, shape {L2.W.shape})
    W_fc, b_fc         (classifier, shape {self.fc_W.shape})
""")
