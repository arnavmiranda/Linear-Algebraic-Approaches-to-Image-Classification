"""
All visualizations for the Envision project.

Figures produced:
  1. filters_layer{n}.png           -  raw learned convolutional filters
  2. eigenfilters_layer{n}.png      -  right singular vectors (V columns)
  3. singular_spectrum.png          -  sigma_i bar chart + cumulative energy
  4. lowrank_recon_layer{n}.png     -  progressive rank-k filter reconstruction
  5. svd_identity_layer{n}.png      -  W vs U Sigma V^T vs diff (should be ~= 0)
  6. relu_subspace.png              -  activation patterns, distributions, U-coords
  7. layer_composition.png          -  singular value chain across both layers
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

OUTDIR = 'outputs'


def _save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved -> {path}")
    return path


# -- Figure 1: raw filter weights --------------------------------------------

def plot_filters(W, title, filename):
    """
    W : (out_ch, dim)   -  dim must be a perfect square (kH == kW)
    Shows each filter as a 2-D heatmap.
    """
    out_ch = W.shape[0]
    k      = int(round(W.shape[1] ** 0.5))   # patch size (e.g. 3 for 9-dim)
    ncols  = min(out_ch, 8)
    nrows  = (out_ch + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 1.6, nrows * 1.6))
    axes = np.array(axes).reshape(nrows, ncols)
    vmax = np.abs(W).max()

    for i in range(out_ch):
        r, c = divmod(i, ncols)
        axes[r, c].imshow(W[i].reshape(k, k), cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax)
        axes[r, c].set_title(f'f{i}', fontsize=7)
        axes[r, c].axis('off')

    for i in range(out_ch, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis('off')

    fig.suptitle(title, fontsize=10, y=1.01)
    plt.tight_layout()
    return _save(fig, filename)


# -- Figure 2: eigenfilters (right singular vectors) -------------------------

def plot_eigenfilters(Vt, s, title, filename):
    """
    Vt : (r, dim)   -  rows are right singular vectors v_i^T
    s  : (r,)       -  singular values
    """
    r    = Vt.shape[0]
    k    = int(round(Vt.shape[1] ** 0.5))
    vmax = np.abs(Vt).max()

    fig, axes = plt.subplots(2, r, figsize=(r * 1.7, 3.6))
    if r == 1:
        axes = axes.reshape(2, 1)

    cum_e = np.cumsum(s**2) / (s**2).sum()

    for i in range(r):
        # top row: eigenfilter image
        axes[0, i].imshow(Vt[i].reshape(k, k), cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'v{i+1}\nsigma={s[i]:.3f}', fontsize=7)

        # bottom row: bar showing how much energy this component carries
        axes[1, i].bar([0], [s[i]**2 / (s**2).sum() * 100],
                       color='steelblue', width=0.6)
        axes[1, i].set_ylim(0, 100)
        axes[1, i].set_xticks([])
        axes[1, i].set_ylabel('energy %' if i == 0 else '')
        axes[1, i].tick_params(labelsize=6)
        axes[1, i].set_title(f'{cum_e[i]*100:.0f}%\ncum', fontsize=6)

    fig.suptitle(title, fontsize=10, y=1.02)
    plt.tight_layout()
    return _save(fig, filename)


# -- Figure 3: singular value spectrum ---------------------------------------

def plot_spectrum(s1, s2, filename='singular_spectrum.png'):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, s, lbl in zip(axes, [s1, s2], ['Layer 1', 'Layer 2']):
        x = np.arange(len(s))
        ax.bar(x, s, color='steelblue', edgecolor='navy', alpha=0.85, width=0.7)
        ax.set_xlabel('Singular value index  i', fontsize=9)
        ax.set_ylabel('sigma_i', fontsize=11)
        ax.set_title(f'{lbl}   -   Singular Value Spectrum', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{i+1}' for i in x], fontsize=7)
        ax.grid(axis='y', alpha=0.3)

        ax2 = ax.twinx()
        cum = np.cumsum(s**2) / (s**2).sum() * 100
        ax2.plot(x, cum, 'r--o', markersize=5, linewidth=1.5,
                 label='Cumulative energy %')
        ax2.axhline(95, color='orange', linestyle=':', linewidth=1, label='95%')
        ax2.axhline(99, color='red',    linestyle=':', linewidth=1, label='99%')
        ax2.set_ylabel('Cumulative energy (%)', color='crimson', fontsize=9)
        ax2.set_ylim(0, 107)
        ax2.tick_params(axis='y', labelcolor='crimson')
        ax2.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    return _save(fig, filename)


# -- Figure 4: low-rank reconstruction ---------------------------------------

def plot_lowrank_recon(W, U, s, Vt, title, filename, ks=None):
    """
    Show how rank-k reconstruction of filters improves as k increases.
    """
    from .svd_analysis import low_rank_approx
    if ks is None:
        ks = list(range(1, min(W.shape[0], 5) + 1))

    out_ch = W.shape[0]
    k_sz   = int(round(W.shape[1] ** 0.5))
    ncols  = len(ks) + 1
    vmax   = np.abs(W).max()

    fig, axes = plt.subplots(out_ch, ncols,
                             figsize=(ncols * 1.6, out_ch * 1.5))
    if out_ch == 1:
        axes = axes.reshape(1, ncols)

    for i in range(out_ch):
        axes[i, 0].imshow(W[i].reshape(k_sz, k_sz),
                          cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original\nW', fontsize=8)

        for j, k in enumerate(ks):
            W_k = low_rank_approx(U, s, Vt, k)
            cum_e = np.cumsum(s[:k]**2).sum() / (s**2).sum()
            axes[i, j+1].imshow(W_k[i].reshape(k_sz, k_sz),
                                cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[i, j+1].axis('off')
            if i == 0:
                axes[i, j+1].set_title(f'Rank-{k}\n({cum_e*100:.0f}%)', fontsize=8)

    fig.suptitle(title, fontsize=10, y=1.01)
    plt.tight_layout()
    return _save(fig, filename)


# -- Figure 5: SVD identity W = U Sigma V^T --------------------------------------

def plot_svd_identity(W, U, s, Vt, layer_name, filename):
    W_rec = (U * s) @ Vt
    diff  = W - W_rec
    vmax  = np.abs(W).max()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    titles = [
        f'Original  W\n{W.shape}',
        f'Reconstructed  U.Sigma.V^T\nmax err = {np.abs(diff).max():.2e}',
        f'Difference  W - USigmaV^T\n(should be ~= 0)',
    ]
    mats   = [W, W_rec, diff]
    cmaps  = ['RdBu_r', 'RdBu_r', 'RdBu_r']

    for ax, mat, ttl, cmap in zip(axes, mats, titles, cmaps):
        im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel('input dims  (in_ch . kH . kW)', fontsize=8)
        ax.set_ylabel('output channels', fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'{layer_name}:  W  =  U . Sigma . V^T  (SVD identity)', fontsize=11)
    plt.tight_layout()
    return _save(fig, filename)


# -- Figure 6: ReLU subspace analysis ----------------------------------------

def plot_relu_subspace(pre, post, coords_pre, coords_post, filename='relu_subspace.png'):
    """
    pre, post       : (N, out_ch)   -  pre- and post-ReLU activations
    coords_pre/post : (N, out_ch)   -  projections onto left singular vectors
    """
    fig = plt.figure(figsize=(16, 4.5))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    #  -  panel 1: binary activation heatmap
    ax1 = fig.add_subplot(gs[0])
    n_show = min(80, len(pre))
    im = ax1.imshow((pre[:n_show] > 0).astype(float).T,
                    cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax1.set_xlabel('Patch sample', fontsize=8)
    ax1.set_ylabel('Filter index', fontsize=8)
    ax1.set_title('ReLU activation mask\n(green=active, red=dead)', fontsize=8)
    plt.colorbar(im, ax=ax1, fraction=0.05)

    #  -  panel 2: activation value distributions
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(pre.ravel(),  bins=60, density=True, alpha=0.7,
             color='steelblue', label='pre-ReLU')
    ax2.hist(post.ravel(), bins=60, density=True, alpha=0.7,
             color='coral',     label='post-ReLU')
    ax2.axvline(0, color='k', linewidth=1)
    ax2.set_xlabel('Activation value', fontsize=8)
    ax2.set_ylabel('Density', fontsize=8)
    ax2.set_title('Distribution before/after ReLU', fontsize=8)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    #  -  panel 3: scatter in first 2 left-singular-vector dims
    ax3 = fig.add_subplot(gs[2])
    ax3.scatter(coords_pre[:, 0],  coords_pre[:, 1],  s=4,  alpha=0.4,
                color='steelblue', label='pre-ReLU')
    ax3.scatter(coords_post[:, 0], coords_post[:, 1], s=4,  alpha=0.4,
                color='coral',     label='post-ReLU')
    ax3.axhline(0, color='k', linewidth=0.5)
    ax3.axvline(0, color='k', linewidth=0.5)
    ax3.set_xlabel('U[:,0]  coord', fontsize=8)
    ax3.set_ylabel('U[:,1]  coord', fontsize=8)
    ax3.set_title('Projection onto top-2\nleft singular vectors', fontsize=8)
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)

    #  -  panel 4: per-filter activation rate bar
    ax4 = fig.add_subplot(gs[3])
    act_rate = (pre > 0).mean(axis=0) * 100
    ax4.barh(range(len(act_rate)), act_rate, color='mediumseagreen', edgecolor='k', linewidth=0.5)
    ax4.set_xlabel('Activation rate (%)', fontsize=8)
    ax4.set_ylabel('Filter index', fontsize=8)
    ax4.set_title('Per-filter\nactivation rate', fontsize=8)
    ax4.set_xlim(0, 100)
    ax4.grid(axis='x', alpha=0.3)
    ax4.set_yticks(range(len(act_rate)))

    fig.suptitle('ReLU creates piecewise-linear subspace partitions', fontsize=10, y=1.02)
    return _save(fig, filename)


# -- Figure 7: two-layer singular value chain --------------------------------

def plot_layer_composition(s1, s2, filename='layer_composition.png'):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # L1 spectrum
    axes[0].bar(range(len(s1)), s1, color='steelblue', edgecolor='navy', alpha=0.85)
    axes[0].set_title('Layer 1  sigma_i', fontsize=10)
    axes[0].set_xlabel('i')
    axes[0].set_ylabel('sigma_i')
    axes[0].grid(axis='y', alpha=0.3)

    # L2 spectrum
    axes[1].bar(range(len(s2)), s2, color='darkorange', edgecolor='saddlebrown', alpha=0.85)
    axes[1].set_title('Layer 2  sigma_i', fontsize=10)
    axes[1].set_xlabel('i')
    axes[1].set_ylabel('sigma_i')
    axes[1].grid(axis='y', alpha=0.3)

    # combined energy flow
    ax = axes[2]
    total1 = np.sum(s1**2)
    total2 = np.sum(s2**2)
    ax.barh([1], [total1], color='steelblue',  alpha=0.8, label='Layer 1  Sigma sigma_i^2')
    ax.barh([0], [total2], color='darkorange', alpha=0.8, label='Layer 2  Sigma sigma_i^2')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Layer 2', 'Layer 1'])
    ax.set_xlabel('Total spectral energy  (Sigma sigma_i^2  =  ||W||^2_F)', fontsize=8)
    ax.set_title('Spectral energy per layer\n(Frobenius norm^2)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return _save(fig, filename)


# -- Figure 8: top patches per eigenfilter -----------------------------------

def plot_top_patches(top_patches_dict, Vt, s, filename='top_patches_eigenfilters.png'):
    """
    For each eigenfilter v_i, show the top-6 patches with highest positive
    projection and top-6 with most negative projection.
    """
    r      = Vt.shape[0]
    n_top  = len(top_patches_dict[0]['pos'])
    k      = int(round(Vt.shape[1] ** 0.5))
    ncols  = n_top * 2 + 1
    nrows  = r

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.1, nrows * 1.2))
    if nrows == 1:
        axes = axes.reshape(1, ncols)

    vmax_ef = np.abs(Vt).max()

    for i in range(r):
        mid = n_top
        axes[i, mid].imshow(Vt[i].reshape(k, k), cmap='RdBu_r',
                            vmin=-vmax_ef, vmax=vmax_ef)
        axes[i, mid].set_title(f'v{i+1}\nσ={s[i]:.2f}', fontsize=6)
        axes[i, mid].axis('off')

        pos_patches = top_patches_dict[i]['pos']
        neg_patches = top_patches_dict[i]['neg']
        vmax_p = max(np.abs(pos_patches).max(), np.abs(neg_patches).max())

        for j in range(n_top):
            col = mid - 1 - j
            axes[i, col].imshow(pos_patches[j].reshape(k, k),
                                cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p)
            axes[i, col].axis('off')
            if i == 0:
                axes[i, col].set_title(f'+{j+1}', fontsize=6)

            col = mid + 1 + j
            axes[i, col].imshow(neg_patches[j].reshape(k, k),
                                cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p)
            axes[i, col].axis('off')
            if i == 0:
                axes[i, col].set_title(f'-{j+1}', fontsize=6)

    fig.text(0.18, 1.01, 'High positive activation', ha='center', fontsize=8, color='navy')
    fig.text(0.50, 1.01, 'Eigenfilter', ha='center', fontsize=8, color='black')
    fig.text(0.82, 1.01, 'High negative activation', ha='center', fontsize=8, color='crimson')
    fig.suptitle('MNIST patches that maximally activate each eigenfilter (V rows)',
                 fontsize=10, y=1.06)
    plt.tight_layout()
    return _save(fig, filename)


# -- Figure 9: per-class eigenfilter response heatmap -----------------------

def plot_class_eigenfilter_heatmap(response, filename='class_eigenfilter_heatmap.png'):
    """response : (10, r)  — mean |projection| per class per eigenfilter"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             gridspec_kw={'width_ratios': [2, 1]})

    ax = axes[0]
    im = ax.imshow(response, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(response.shape[1]))
    ax.set_xticklabels([f'v{i+1}' for i in range(response.shape[1])], fontsize=8)
    ax.set_yticks(range(10))
    ax.set_yticklabels([str(c) for c in range(10)], fontsize=9)
    ax.set_xlabel('Eigenfilter  (V row)', fontsize=9)
    ax.set_ylabel('Digit class', fontsize=9)
    ax.set_title('Mean |projection| of each digit class\nonto each eigenfilter direction', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label='mean |Vᵀp|')

    ax2 = axes[1]
    best_ef = np.argmax(response, axis=1)
    colors  = plt.cm.tab10(np.linspace(0, 1, 10))
    ax2.barh(range(10), response[range(10), best_ef],
             color=colors, edgecolor='k', linewidth=0.5)
    ax2.set_yticks(range(10))
    ax2.set_yticklabels([f'digit {c}  →  v{best_ef[c]+1}' for c in range(10)], fontsize=7)
    ax2.set_xlabel('Response magnitude', fontsize=8)
    ax2.set_title('Strongest eigenfilter\nper digit class', fontsize=9)
    ax2.grid(axis='x', alpha=0.3)

    fig.suptitle('How each digit class engages the learned eigenfilters', fontsize=11, y=1.02)
    plt.tight_layout()
    return _save(fig, filename)


# -- Figure 10: U-space class scatter ----------------------------------------

def plot_uspace_scatter(coords, labels, filename='uspace_class_scatter.png'):
    """coords : (N, r), labels : (N,)"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap   = plt.cm.tab10
    colors = [cmap(c / 9) for c in range(10)]

    for ax_idx, (d1, d2) in enumerate([(0, 1), (2, 3)]):
        ax = axes[ax_idx]
        if coords.shape[1] <= d2:
            ax.text(0.5, 0.5, 'Not enough dimensions', ha='center', va='center')
            ax.axis('off')
            continue
        for c in range(10):
            mask = labels == c
            ax.scatter(coords[mask, d1], coords[mask, d2],
                       s=6, alpha=0.4, color=colors[c], label=str(c))
        ax.set_xlabel(f'U col {d1+1}', fontsize=9)
        ax.set_ylabel(f'U col {d2+1}', fontsize=9)
        ax.set_title(f'Digits in U-space  (dims {d1+1} & {d2+1})', fontsize=9)
        ax.grid(alpha=0.2)
        if ax_idx == 1:
            ax.legend(title='Digit', fontsize=7, markerscale=2,
                      loc='upper right', ncol=2)

    fig.suptitle('MNIST test images projected onto Layer-1 left singular vectors (U)\n'
                 'Each point is one image; colour = digit class', fontsize=10)
    plt.tight_layout()
    return _save(fig, filename)


# -- Figure 11: data SVD vs weight SVD comparison ----------------------------

def plot_data_vs_weight_svd(s_data, s_W, alignment,
                             filename='data_vs_weight_svd.png'):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    r = len(s_W)
    x = np.arange(r)

    ax = axes[0]
    ax.bar(x - 0.2, s_data[:r] / s_data[:r].max(), width=0.35,
           color='steelblue', alpha=0.85, label='Data SVD (normalised)')
    ax.bar(x + 0.2, s_W / s_W.max(),               width=0.35,
           color='darkorange', alpha=0.85, label='Weight SVD (normalised)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}' for i in x], fontsize=8)
    ax.set_xlabel('Component index', fontsize=9)
    ax.set_ylabel('Normalised singular value', fontsize=9)
    ax.set_title('Singular value profiles:\ndata PCA  vs  learned weights', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    bar_colors = ['#2ecc71' if a > 0.7 else '#e67e22' if a > 0.4 else '#e74c3c'
                  for a in alignment]
    ax.bar(x, alignment, color=bar_colors, edgecolor='k', linewidth=0.5)
    ax.axhline(0.7, color='orange', linestyle=':', linewidth=1, label='0.7 threshold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'v{i+1}' for i in x], fontsize=8)
    ax.set_xlabel('Eigenfilter index', fontsize=9)
    ax.set_ylabel('|cos θ|  (alignment)', fontsize=9)
    ax.set_title('Alignment between weight eigenfilters\nand data PCA directions', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[2]
    ax.axis('off')
    mean_align = alignment.mean()
    interp = (
        "High alignment (green) means the\n"
        "network's eigenfilter matches the\n"
        "dominant direction in the data.\n\n"
        "Low alignment (red) means the\n"
        "network learned directions the\n"
        "raw data PCA did not find —\n"
        "i.e. class-discriminative patterns\n"
        "that variance alone misses.\n\n"
        f"Mean alignment: {mean_align:.3f}\n\n"
        + ("The network closely follows the\ndata's principal components."
           if mean_align > 0.7 else
           "The network diverges from PCA —\ndiscrimination > reconstruction.")
    )
    ax.text(0.05, 0.95, interp, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('Interpretation', fontsize=9)

    fig.suptitle("Do learned eigenfilters match the data's own principal components?",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    return _save(fig, filename)
