"""
Dataset-linked SVD analysis.

Connects the learned SVD components {U, Sigma, V} back to the original
MNIST dataset, answering three questions:

  1. What do the eigenfilters (V rows) actually look for in real images?
     -> Find the MNIST patches that maximally activate each v_i direction.

  2. How does each digit class respond to each eigenfilter?
     -> Per-class mean projection onto V-space: a (10 x r) heatmap.
     -> Reveals which eigenfilters are class-discriminative.

  3. How does the learned output basis (U) organise the digit classes?
     -> Project every test image's Layer 1 activations onto U columns.
     -> 2D scatter in the top-2 U directions, coloured by class.

  4. Does gradient descent rediscover the data's own principal components?
     -> Compare the singular values of the raw MNIST patch matrix with
        those of the learned weight matrix W.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .svd_analysis import batch_im2col


# ---------------------------------------------------------------------------
#  Helper: extract all patches + labels from a loader
# ---------------------------------------------------------------------------

def extract_patches_and_labels(loader, model, n_images=2000):
    """
    Run n_images through the model, collecting:
      patches   : (N*26*26, 9)   - all Layer-1 input patches
      labels    : (N*26*26,)     - digit label replicated per patch
      pre_relu1 : (N*26*26, 8)   - Layer-1 pre-ReLU activations
      img_acts  : (N, 8)         - mean post-ReLU activation per image (for U-scatter)
      img_labels: (N,)           - per-image labels
    """
    all_patches, all_labels = [], []
    all_pre, img_acts, img_labels = [], [], []

    collected = 0
    for images, labels in loader:
        if collected >= n_images:
            break
        batch = images[:min(len(images), n_images - collected)]
        lbls  = labels[:len(batch)]
        collected += len(batch)

        imgs_np = batch.numpy()[:, 0]   # (B, 28, 28)

        with torch.no_grad():
            acts = model.get_activations(batch)

        pre = acts['pre_relu1'].numpy()              # (B, 8, 26, 26)
        B   = len(batch)

        patches = batch_im2col(imgs_np, kH=3, kW=3)  # (B*676, 9)
        n_p     = pre.shape[2] * pre.shape[3]        # 676

        pre_flat = pre.transpose(0, 2, 3, 1).reshape(B * n_p, 8)  # (B*676, 8)
        lbl_rep  = np.repeat(lbls.numpy(), n_p)                    # (B*676,)

        all_patches.append(patches)
        all_labels.append(lbl_rep)
        all_pre.append(pre_flat)

        # Per-image: mean activation across spatial positions
        mean_act = acts['post_relu1'].numpy().mean(axis=(2, 3))   # (B, 8)
        img_acts.append(mean_act)
        img_labels.append(lbls.numpy())

    return (
        np.vstack(all_patches),
        np.concatenate(all_labels),
        np.vstack(all_pre),
        np.vstack(img_acts),
        np.concatenate(img_labels),
    )


# ---------------------------------------------------------------------------
#  Analysis 1 — Top patches per eigenfilter
# ---------------------------------------------------------------------------

def top_patches_per_eigenfilter(patches, Vt, n_top=6):
    """
    For each eigenfilter v_i (row i of Vt), find the n_top patches that
    project most strongly (positive and negative) onto v_i.

    Returns dict: { i : {'pos': (n_top, 9), 'neg': (n_top, 9),
                         'pos_scores': (n_top,), 'neg_scores': (n_top,) } }
    """
    projections = patches @ Vt.T   # (N, r)  — coordinate in each v_i direction
    result = {}
    for i in range(Vt.shape[0]):
        scores = projections[:, i]
        pos_idx = np.argsort(scores)[-n_top:][::-1]
        neg_idx = np.argsort(scores)[:n_top]
        result[i] = {
            'pos':        patches[pos_idx],
            'neg':        patches[neg_idx],
            'pos_scores': scores[pos_idx],
            'neg_scores': scores[neg_idx],
        }
    return result


# ---------------------------------------------------------------------------
#  Analysis 2 — Per-class eigenfilter response
# ---------------------------------------------------------------------------

def per_class_eigenfilter_response(patches, labels, Vt):
    """
    For each digit class c and each eigenfilter i, compute the mean absolute
    projection of class-c patches onto v_i.

    Returns response matrix of shape (10, r).
    """
    projections = patches @ Vt.T   # (N, r)
    r = Vt.shape[0]
    response = np.zeros((10, r))
    for c in range(10):
        mask = labels == c
        if mask.sum() > 0:
            response[c] = np.abs(projections[mask]).mean(axis=0)
    return response


# ---------------------------------------------------------------------------
#  Analysis 3 — U-space class scatter
# ---------------------------------------------------------------------------

def uspace_class_projections(img_acts, img_labels, U):
    """
    Project per-image mean activations onto the left singular vectors of W1.
    img_acts : (N, out_ch)   — mean post-ReLU activation per image
    U        : (out_ch, r)

    Returns coords (N, r) and labels (N,).
    """
    coords = img_acts @ U   # (N, r)
    return coords, img_labels


# ---------------------------------------------------------------------------
#  Analysis 4 — Data SVD vs weight SVD
# ---------------------------------------------------------------------------

def data_svd_comparison(patches, W, n_components=8):
    """
    Compute SVD of the patch data matrix and compare its singular values
    and right singular vectors with those of the weight matrix W.

    patches : (N, 9)   — raw MNIST patches
    W       : (8, 9)   — learned weight matrix

    Returns:
      s_data  : (r,)   singular values of patch matrix (top n_components)
      Vt_data : (r, 9) right singular vectors of patch data
      alignment : (r,) cosine similarity between each W v_i and data v_i
    """
    # Centre patches
    patches_c = patches - patches.mean(axis=0)

    # Thin SVD of data matrix (N x 9) — right singular vectors are PCA directions
    _, s_data, Vt_data = np.linalg.svd(patches_c, full_matrices=False)
    s_data   = s_data[:n_components]
    Vt_data  = Vt_data[:n_components]

    # Weight SVD
    _, s_W, Vt_W = np.linalg.svd(W, full_matrices=False)

    # Alignment: |cos theta| between each pair of singular vectors
    alignment = np.abs(np.einsum('ij,ij->i', Vt_W, Vt_data[:len(s_W)]))

    print("\n" + "=" * 62)
    print("  Data SVD  vs  Weight SVD")
    print("=" * 62)
    print(f"\n  {'':4}  {'sigma_data':>12}  {'sigma_W':>10}  {'|cos(v_data, v_W)|':>20}")
    print(f"  {'-'*52}")
    for i in range(len(s_W)):
        print(f"  {i+1:>4}  {s_data[i]:>12.4f}  {s_W[i]:>10.4f}  {alignment[i]:>20.4f}")

    print(f"\n  Mean alignment |cos theta| = {alignment.mean():.4f}")
    print(f"  (1.0 = weight eigenfilters perfectly aligned with data PCA directions)")
    print(f"  (0.0 = completely unrelated)")

    return s_data, Vt_data, alignment


# ---------------------------------------------------------------------------
#  Summary printer
# ---------------------------------------------------------------------------

def print_dataset_summary(response, img_labels):
    print("\n" + "=" * 62)
    print("  Per-Class Eigenfilter Response (mean |projection|)")
    print("=" * 62)
    r = response.shape[1]
    header = f"  {'class':>6}" + "".join(f"   v{i+1:>2}" for i in range(r))
    print(header)
    print(f"  {'-' * (8 + r*7)}")
    for c in range(10):
        row = f"  {c:>6}" + "".join(f"  {response[c, i]:>5.3f}" for i in range(r))
        print(row)

    # Most discriminative eigenfilter per class
    print(f"\n  Most responsive eigenfilter per digit:")
    for c in range(10):
        best = np.argmax(response[c])
        print(f"    digit {c}  ->  v{best+1}  ({response[c, best]:.3f})")
