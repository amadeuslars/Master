"""
Plot PPO training metrics from TensorBoard event files.

Reads SB3 PPO event files for up to 4 trained agents and generates:
  1. Per-agent metric dashboards (all metrics in one figure)
  2. Cross-agent comparison figures for key metrics

Usage:
    python plots/plot_training.py
    python plots/plot_training.py --output plots/training/
    python plots/plot_training.py --agents 5310_all_files_agent new_rew_all_files_agent
"""

import argparse
import os
import glob
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

# --- Default agent configurations ---
DEFAULT_AGENTS = {
    '5310_all_files_agent': {
        'label': '5310 All Files',
        'color': '#1f77b4',
    },
    '5310_type1_files_agent': {
        'label': '5310 Type-1',
        'color': '#ff7f0e',
    },
    'new_rew_all_files_agent': {
        'label': 'New Reward All',
        'color': '#2ca02c',
    },
    'new_rew_type1_files_agent': {
        'label': 'New Reward Type-1',
        'color': '#d62728',
    },
}

# Key metrics to compare across agents
COMPARISON_METRICS = [
    'rollout/ep_rew_mean',
    'train/entropy_loss',
    'train/loss',
    'train/approx_kl',
    'train/explained_variance',
]

# All metrics for per-agent dashboards
ALL_METRICS = [
    'rollout/ep_rew_mean',
    'rollout/ep_len_mean',
    'train/approx_kl',
    'train/clip_fraction',
    'train/entropy_loss',
    'train/explained_variance',
    'train/loss',
    'train/policy_gradient_loss',
    'train/value_loss',
]

METRIC_LABELS = {
    'rollout/ep_rew_mean': 'Episode Reward (mean)',
    'rollout/ep_len_mean': 'Episode Length (mean)',
    'train/approx_kl': 'Approx. KL Divergence',
    'train/clip_fraction': 'Clip Fraction',
    'train/entropy_loss': 'Entropy Loss',
    'train/explained_variance': 'Explained Variance',
    'train/loss': 'Total Loss',
    'train/policy_gradient_loss': 'Policy Gradient Loss',
    'train/value_loss': 'Value Loss',
}


def _ema(values, alpha=0.1):
    """Exponential moving average for smoothing."""
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def load_events(event_dir):
    """
    Load scalar data from TensorBoard event files.

    Returns dict: tag → {'steps': array, 'values': array}
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    # Find event file
    event_files = glob.glob(os.path.join(event_dir, 'events.out.tfevents.*'))
    if not event_files:
        # Try one level deeper (PPO_VRPTW_1/)
        event_files = glob.glob(os.path.join(event_dir, '*', 'events.out.tfevents.*'))
    if not event_files:
        print(f"No event files found in {event_dir}")
        return {}

    ea = EventAccumulator(os.path.dirname(event_files[0]))
    ea.Reload()

    data = {}
    for tag in ea.Tags().get('scalars', []):
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        data[tag] = {'steps': steps, 'values': values}

    return data


def plot_agent_dashboard(data, agent_name, agent_label, output_path=None):
    """
    Plot all metrics for a single agent as a grid of subplots.
    """
    available = [m for m in ALL_METRICS if m in data]
    n = len(available)
    if n == 0:
        print(f"No metrics found for {agent_name}")
        return None

    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, metric in enumerate(available):
        ax = axes[i]
        steps = data[metric]['steps']
        values = data[metric]['values']

        # Plot raw (faint) + EMA smoothed
        ax.plot(steps, values, alpha=0.2, color='#1f77b4', linewidth=0.5)
        smoothed = _ema(values, alpha=0.05)
        ax.plot(steps, smoothed, color='#1f77b4', linewidth=1.5)

        ax.set_xlabel('Timestep')
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'Training Metrics — {agent_label}', fontsize=13, y=1.01)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)
        print(f"Saved: {output_path}")
    return fig


def plot_comparison(all_data, agents_config, metric, output_path=None):
    """
    Compare a single metric across multiple agents.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    for agent_name, config in agents_config.items():
        if agent_name not in all_data or metric not in all_data[agent_name]:
            continue
        steps = all_data[agent_name][metric]['steps']
        values = all_data[agent_name][metric]['values']
        smoothed = _ema(values, alpha=0.05)

        ax.plot(steps, values, alpha=0.1, color=config['color'], linewidth=0.5)
        ax.plot(steps, smoothed, color=config['color'], linewidth=1.5,
                label=config['label'])

    ax.set_xlabel('Timestep')
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(METRIC_LABELS.get(metric, metric))
    ax.legend(loc='best', framealpha=0.8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)
        print(f"Saved: {output_path}")
    return fig


def plot_comparison_grid(all_data, agents_config, output_path=None):
    """
    Multi-panel comparison of key metrics across agents.
    """
    available = [m for m in COMPARISON_METRICS if any(
        m in all_data.get(a, {}) for a in agents_config
    )]
    n = len(available)
    if n == 0:
        print("No comparison metrics available")
        return None

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, metric in enumerate(available):
        ax = axes[i]
        for agent_name, config in agents_config.items():
            if agent_name not in all_data or metric not in all_data[agent_name]:
                continue
            steps = all_data[agent_name][metric]['steps']
            values = all_data[agent_name][metric]['values']
            smoothed = _ema(values, alpha=0.05)
            ax.plot(steps, smoothed, color=config['color'], linewidth=1.3,
                    label=config['label'])

        ax.set_xlabel('Timestep')
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(loc='best', fontsize=7, framealpha=0.8)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Training Comparison Across Agents', fontsize=13, y=1.01)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)
        print(f"Saved: {output_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot PPO training metrics from TensorBoard events')
    parser.add_argument('--logs-dir', type=str, default=None,
                        help='Root logs directory (default: auto-detect)')
    parser.add_argument('--agents', type=str, nargs='+', default=None,
                        help='Agent names to plot (default: all 4)')
    parser.add_argument('--output', type=str, default='plots/training/',
                        help='Output directory for figures')
    args = parser.parse_args()

    # Auto-detect logs directory
    if args.logs_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        logs_dir = os.path.join(project_root, 'logs')
    else:
        logs_dir = args.logs_dir

    os.makedirs(args.output, exist_ok=True)

    # Filter agents
    agents_config = DEFAULT_AGENTS
    if args.agents:
        agents_config = {k: v for k, v in DEFAULT_AGENTS.items() if k in args.agents}

    # Load all event data
    all_data = {}
    for agent_name in agents_config:
        event_dir = os.path.join(logs_dir, agent_name, 'PPO_VRPTW_1')
        if not os.path.isdir(event_dir):
            event_dir = os.path.join(logs_dir, agent_name)
        if not os.path.isdir(event_dir):
            print(f"Skipping {agent_name}: directory not found")
            continue

        print(f"Loading events for {agent_name}...")
        data = load_events(event_dir)
        if data:
            all_data[agent_name] = data
            print(f"  Found {len(data)} metrics, "
                  f"{max(len(v['steps']) for v in data.values())} max datapoints")

    if not all_data:
        print("No data loaded. Check logs directory.")
        return

    # Per-agent dashboards
    for agent_name, data in all_data.items():
        config = agents_config[agent_name]
        plot_agent_dashboard(
            data, agent_name, config['label'],
            output_path=os.path.join(args.output, f'{agent_name}_dashboard.png')
        )

    # Cross-agent comparison grid
    if len(all_data) > 1:
        plot_comparison_grid(
            all_data, agents_config,
            output_path=os.path.join(args.output, 'comparison_grid.png')
        )

        # Individual comparison plots for key metrics
        for metric in COMPARISON_METRICS:
            safe_name = metric.replace('/', '_')
            plot_comparison(
                all_data, agents_config, metric,
                output_path=os.path.join(args.output, f'comparison_{safe_name}.png')
            )

    plt.close('all')
    print(f"\nAll figures saved to {args.output}")


if __name__ == '__main__':
    main()
