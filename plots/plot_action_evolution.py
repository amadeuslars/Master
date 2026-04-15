"""
Plot action probability evolution for ALNS-RRT and DRLH solvers.

Generates three figure types:
  Fig A — Family-level probabilities vs iteration (destroy, repair, bucket)
  Fig B — Action heatmap (30 actions × iteration bins)
  Fig C — Top-k action trajectories

Usage:
    # After running solvers that return history dicts:
    from plots.plot_action_evolution import plot_family_probs, plot_action_heatmap, plot_topk_trajectories

    # Or run standalone with saved history .npz files:
    python plots/plot_action_evolution.py --history path/to/history.npz --output plots/
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- Thesis-quality defaults ---
mpl.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# --- Action encoding ---
DESTROY_NAMES = ['random', 'worst', 'cluster']
BUCKET_NAMES = ['xs', 'sm', 'md', 'lg', 'xl']
REPAIR_NAMES = ['greedy', 'regret']
NUM_ACTIONS = 30


def _build_action_labels():
    """Build canonical 30 action labels in index order."""
    labels = []
    for d in DESTROY_NAMES:
        for s in BUCKET_NAMES:
            for r in REPAIR_NAMES:
                labels.append(f"{d}_{s}_{r}")
    return labels


def _decode_action(idx):
    """Decode composite action index → (d_idx, s_idx, r_idx)."""
    r_idx = idx % 2
    s_idx = (idx // 2) % 5
    d_idx = idx // 10
    return d_idx, s_idx, r_idx


def _get_probs_array(history):
    """
    Get (N, 30) probability array from history.
    For RRT: normalize action_weights per row.
    For DRLH: use policy_probs directly.
    """
    if history['algorithm'] == 'DRLH':
        return history['policy_probs']
    else:
        weights = history['action_weights']
        row_sums = weights.sum(axis=1, keepdims=True)
        return weights / row_sums


def _rolling_mean(arr, window):
    """Rolling mean along axis 0 for 1D or 2D array."""
    if arr.ndim == 1:
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode='valid')
    else:
        out = np.zeros((arr.shape[0] - window + 1, arr.shape[1]))
        for j in range(arr.shape[1]):
            kernel = np.ones(window) / window
            out[:, j] = np.convolve(arr[:, j], kernel, mode='valid')
        return out


def _marginalize(probs):
    """
    Marginalize (N, 30) action probs into family-level probs.

    Returns dict with:
      'destroy': (N, 3) — sum over matching destroy ops
      'repair':  (N, 2) — sum over matching repair ops
      'bucket':  (N, 5) — sum over matching buckets
    """
    N = probs.shape[0]
    destroy_probs = np.zeros((N, 3))
    repair_probs = np.zeros((N, 2))
    bucket_probs = np.zeros((N, 5))

    for idx in range(NUM_ACTIONS):
        d_idx, s_idx, r_idx = _decode_action(idx)
        destroy_probs[:, d_idx] += probs[:, idx]
        bucket_probs[:, s_idx] += probs[:, idx]
        repair_probs[:, r_idx] += probs[:, idx]

    return {
        'destroy': destroy_probs,
        'repair': repair_probs,
        'bucket': bucket_probs,
    }


# ====================================================================
#  Fig A — Family-level probabilities
# ====================================================================

def plot_family_probs(history, window=100, title_prefix='', output_path=None):
    """
    Plot marginalized family-level probabilities vs iteration.

    3 subplots: destroy (3 lines), repair (2 lines), bucket (5 lines).
    """
    probs = _get_probs_array(history)
    families = _marginalize(probs)
    algo = history['algorithm']

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

    configs = [
        ('destroy', DESTROY_NAMES, ['#e41a1c', '#377eb8', '#4daf4a']),
        ('repair', REPAIR_NAMES, ['#ff7f00', '#984ea3']),
        ('bucket', BUCKET_NAMES, ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99']),
    ]

    for ax, (family_key, names, colors) in zip(axes, configs):
        data = families[family_key]
        if data.shape[0] > window:
            smoothed = _rolling_mean(data, window)
            x = np.arange(window - 1, data.shape[0])
        else:
            smoothed = data
            x = np.arange(data.shape[0])

        for j, (name, color) in enumerate(zip(names, colors)):
            ax.plot(x, smoothed[:, j], label=name, color=color, linewidth=1.2)

        ax.set_xlabel('Iteration')
        ax.set_title(family_key.capitalize())
        ax.legend(loc='best', framealpha=0.8)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel('Marginal probability')
    fig.suptitle(f'{title_prefix}{algo} — Family-Level Action Probabilities', fontsize=12, y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)
        print(f"Saved: {output_path}")
    return fig


# ====================================================================
#  Fig B — Action heatmap
# ====================================================================

def plot_action_heatmap(history, n_bins=50, title_prefix='', output_path=None):
    """
    Heatmap of 30 actions × iteration bins.
    Rows sorted by mean probability (highest at top).
    """
    probs = _get_probs_array(history)
    algo = history['algorithm']
    labels = history.get('action_labels', _build_action_labels())

    N = probs.shape[0]
    bin_size = max(1, N // n_bins)
    n_actual_bins = N // bin_size

    # Bin probabilities
    binned = np.zeros((NUM_ACTIONS, n_actual_bins))
    for b in range(n_actual_bins):
        start = b * bin_size
        end = start + bin_size
        binned[:, b] = probs[start:end, :].mean(axis=0)

    # Sort by mean probability (descending)
    mean_probs = binned.mean(axis=1)
    sort_idx = np.argsort(mean_probs)[::-1]
    binned_sorted = binned[sort_idx, :]
    labels_sorted = [labels[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(binned_sorted, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    # X-axis: iteration bins
    tick_positions = np.linspace(0, n_actual_bins - 1, min(6, n_actual_bins)).astype(int)
    tick_labels = [str(int(t * bin_size)) for t in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('Iteration')

    # Y-axis: action labels
    ax.set_yticks(range(NUM_ACTIONS))
    ax.set_yticklabels(labels_sorted, fontsize=7)
    ax.set_ylabel('Action (sorted by mean prob)')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Probability')

    ax.set_title(f'{title_prefix}{algo} — Action Probability Heatmap', fontsize=12)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)
        print(f"Saved: {output_path}")
    return fig


# ====================================================================
#  Fig C — Top-k action trajectories
# ====================================================================

def plot_topk_trajectories(history, k=5, window=100, title_prefix='', output_path=None):
    """
    Plot top-k actions by mean probability as individual lines.
    Remaining actions aggregated as shaded 'other' band.
    """
    probs = _get_probs_array(history)
    algo = history['algorithm']
    labels = history.get('action_labels', _build_action_labels())

    mean_probs = probs.mean(axis=0)
    top_indices = np.argsort(mean_probs)[::-1][:k]
    other_indices = np.argsort(mean_probs)[::-1][k:]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    colors = plt.cm.tab10(np.linspace(0, 1, k))

    if probs.shape[0] > window:
        smoothed = _rolling_mean(probs, window)
        x = np.arange(window - 1, probs.shape[0])
    else:
        smoothed = probs
        x = np.arange(probs.shape[0])

    for idx, color in zip(top_indices, colors):
        ax.plot(x, smoothed[:, idx], label=f"{labels[idx]} ({mean_probs[idx]:.3f})",
                color=color, linewidth=1.3)

    # Aggregate 'other'
    other_sum = smoothed[:, other_indices].sum(axis=1)
    ax.fill_between(x, 0, other_sum, alpha=0.15, color='gray', label=f'other ({NUM_ACTIONS - k} actions)')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Probability')
    ax.set_title(f'{title_prefix}{algo} — Top-{k} Action Trajectories', fontsize=12)
    ax.legend(loc='best', fontsize=7, framealpha=0.8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)
        print(f"Saved: {output_path}")
    return fig


# ====================================================================
#  Save / Load utilities
# ====================================================================

def save_history(history, path):
    """Save history dict to .npz file."""
    save_dict = {}
    for k, v in history.items():
        if isinstance(v, np.ndarray):
            save_dict[k] = v
        elif isinstance(v, list) and v and isinstance(v[0], str):
            save_dict[k] = np.array(v, dtype=object)
        elif isinstance(v, str):
            save_dict[k] = np.array(v)
        else:
            save_dict[k] = np.array(v)
    np.savez_compressed(path, **save_dict)
    print(f"Saved history: {path}")


def load_history(path):
    """Load history dict from .npz file."""
    data = np.load(path, allow_pickle=True)
    history = {}
    for k in data.files:
        v = data[k]
        if v.ndim == 0:
            history[k] = str(v)
        elif v.dtype == object:
            history[k] = list(v)
        else:
            history[k] = v
    return history


# ====================================================================
#  CLI
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description='Plot action probability evolution')
    parser.add_argument('--history', type=str, nargs='+', required=True,
                        help='Path(s) to history .npz file(s)')
    parser.add_argument('--output', type=str, default='plots/',
                        help='Output directory for figures')
    parser.add_argument('--window', type=int, default=100,
                        help='Rolling window size for smoothing')
    parser.add_argument('--bins', type=int, default=50,
                        help='Number of iteration bins for heatmap')
    parser.add_argument('--topk', type=int, default=5,
                        help='Number of top actions to show in trajectory plot')
    args = parser.parse_args()

    import os
    os.makedirs(args.output, exist_ok=True)

    for hist_path in args.history:
        history = load_history(hist_path)
        base = os.path.splitext(os.path.basename(hist_path))[0]
        prefix = f"{base} — "

        plot_family_probs(
            history, window=args.window, title_prefix=prefix,
            output_path=os.path.join(args.output, f'{base}_family_probs.png')
        )
        plot_action_heatmap(
            history, n_bins=args.bins, title_prefix=prefix,
            output_path=os.path.join(args.output, f'{base}_heatmap.png')
        )
        plot_topk_trajectories(
            history, k=args.topk, window=args.window, title_prefix=prefix,
            output_path=os.path.join(args.output, f'{base}_topk.png')
        )

    plt.show()


if __name__ == '__main__':
    main()
