"""
SVD analysis of convolutional weight matrices.

Key insight:
  Every Conv2d weight tensor W_raw of shape (out_ch, in_ch, kH, kW) can be
  reshaped into a 2-D matrix  W in R^{out_ch x (in_ch.kH.kW)}.

  SVD gives:  W = U Sigma V^T
    U  in R^{out_ch x out_ch}           -  output feature-space rotation
    Sigma  in R^{out_ch x min(out_ch,dim)}  -  diagonal, singular values sigma_i >= 0
    V  in R^{dim    x dim}              -  input patch-space rotation  (dim = in_ch.kH.kW)

  The columns of V (right singular vectors) are "eigenfilters":
  orthogonal directions in patch space that the layer is most sensitive to,
  ranked by their sigma_i.

  The columns of U (left singular vectors) are the output feature directions
  those eigenfilters map into.
"""

import numpy as np


# --- core decomposition ------------------------------------------------------

def filter_matrix(conv_layer):
    """
    Reshape conv weight tensor -> 2-D matrix (out_ch, in_ch*kH*kW).
    Returns numpy array.
    """
    W = conv_layer.weight.data.numpy()
    return W.reshape(W.shape[0], -1)


def svd_decompose(W):
    """
    Full SVD of W.  Returns U, s, Vt  (numpy, thin-SVD so shapes are:
        U  : (out_ch, r)  where r = min(out_ch, dim)
        s  : (r,)
        Vt : (r, dim)
    """
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    return U, s, Vt


def low_rank_approx(U, s, Vt, k):
    """Reconstruct W using only the top-k singular triplets."""
    return (U[:, :k] * s[:k]) @ Vt[:k, :]


def cumulative_energy(s):
    """Return array of cumulative energy fractions (one per singular value)."""
    sq = s ** 2
    return np.cumsum(sq) / sq.sum()


def effective_rank(s, threshold=0.99):
    """Smallest k such that top-k singular values capture >= threshold of energy."""
    return int(np.argmax(cumulative_energy(s) >= threshold)) + 1


# --- layer-level analysis ----------------------------------------------------

def analyze_layer(conv_layer, name="Layer"):
    """
    Full SVD analysis of one conv layer.
    Prints a report and returns (W, U, s, Vt).
    """
    W = filter_matrix(conv_layer)
    U, s, Vt = svd_decompose(W)

    rank     = np.linalg.matrix_rank(W)
    cond     = s[0] / s[-1] if s[-1] > 1e-12 else float('inf')
    eff99    = effective_rank(s, 0.99)
    eff95    = effective_rank(s, 0.95)
    cum      = cumulative_energy(s)

    print(f"\n{'='*62}")
    print(f"  SVD Analysis  -  {name}")
    print(f"{'='*62}")
    print(f"  Weight matrix shape      : {W.shape}")
    print(f"  True rank                : {rank}  (max possible {min(W.shape)})")
    print(f"  Condition number sigma1/sigma_n  : {cond:.2f}")
    print(f"  Singular values          : {np.round(s, 4)}")
    print()
    print(f"  {'k':>4}  {'sigma_k':>10}  {'energy(k)':>10}  {'cum %':>8}")
    print(f"  {'-'*38}")
    for i, (sv, e) in enumerate(zip(s, cum)):
        marker = " <- 95%" if i + 1 == eff95 else (" <- 99%" if i + 1 == eff99 else "")
        print(f"  {i+1:>4}  {sv:>10.4f}  {sv**2:>10.4f}  {e*100:>7.1f}%{marker}")

    return W, U, s, Vt


# --- patch-level projections -------------------------------------------------

def project_patch(patch, Vt, s):
    """
    Show the 3-step analytical decomposition for one patch.
      patch : 1-D array of length (in_ch*kH*kW)
    Returns:
      z   -  coordinates in V-basis  (input subspace)
      zs  -  z scaled by sigma           (what gets fed into U)
      y   -  U zs = W @ patch        (filter response)
    """
    patch = np.asarray(patch).ravel()
    z  = Vt @ patch           # project onto right singular vectors
    zs = s * z                # scale by singular values
    return z, zs


def im2col(img, kH=3, kW=3):
    """
    Extract all (kHxkW) patches from a single-channel 2-D image.
    Returns array of shape (n_patches, kH*kW)  -  one row per patch.
    """
    H, W = img.shape
    out_H = H - kH + 1
    out_W = W - kW + 1
    patches = np.empty((out_H * out_W, kH * kW), dtype=img.dtype)
    idx = 0
    for i in range(out_H):
        for j in range(out_W):
            patches[idx] = img[i:i+kH, j:j+kW].ravel()
            idx += 1
    return patches


def batch_im2col(imgs_np, kH=3, kW=3):
    """
    imgs_np : (N, H, W)
    Returns : (N * out_H * out_W, kH*kW)
    """
    return np.vstack([im2col(img, kH, kW) for img in imgs_np])


# --- subspace geometry -------------------------------------------------------

def principal_angles(A, B):
    """
    Compute principal angles (degrees) between subspaces spanned by rows of A and B.
    Uses QR -> SVD of cross-gram matrix.
    """
    Q1, _ = np.linalg.qr(A.T)
    Q2, _ = np.linalg.qr(B.T)
    cos_angles = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angles))


def reconstruction_error(W, W_approx, patches):
    """
    Relative Frobenius error when applying W_approx instead of W to patches.
      patches : (N, dim)
    """
    exact = patches @ W.T
    approx = patches @ W_approx.T
    return np.linalg.norm(exact - approx) / (np.linalg.norm(exact) + 1e-12)
