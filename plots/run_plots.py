"""
Tweakable script to generate action evolution plots.

Usage:
    python plots/run_plots.py

To regenerate from saved .npz files without re-running solvers,
set RUN_SOLVER = False and point HISTORY_FILE to the .npz path.
"""

import matplotlib
matplotlib.use('Agg')  # Remove this line to show plots interactively

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plots.plot_action_evolution import (
    plot_family_probs, plot_action_heatmap, plot_topk_trajectories,
    save_history, load_history,
)

# ============================================================
#  SETTINGS — Tweak these
# ============================================================

# --- What to run ---
RUN_SOLVER = True          # True = run solver first, False = load from .npz
SOLVER = 'DRLH'            # 'DRLH' or 'RRT'
CONTEXT = 'case_study'      # 'benchmark' or 'case_study'

# --- Benchmark settings (only used if CONTEXT = 'benchmark') ---
INSTANCE = '/Users/amadeuslinge/UiB/Master/Benchmark/data/homberger_600/C1_6_1.TXT'
MODEL = '/Users/amadeuslinge/UiB/Master/logs/5310_all_files_agent/ppo_vrptw_final.zip'
DETERMINISTIC = True       # DRLH only: True = argmax, False = sample from policy

# --- Case study settings (only used if CONTEXT = 'case_study') ---
DELIVERY_DAY = 'tue'
CUSTOMERS_FILE = 'Case_study/data/customers_alesund_sula_tue.csv'

# --- Load from file (only used if RUN_SOLVER = False) ---
HISTORY_FILE = 'plots/drlh_C1_6_1.npz'

# --- Plot settings ---
WINDOW = 100               # Rolling window for smoothing (1 = raw per-step)
N_BINS = 50                # Number of iteration bins for heatmap
TOP_K = 5                  # Number of top actions in trajectory plot

# --- Output ---
OUTPUT_DIR = 'plots/'
NAME_PREFIX = ''            # e.g. 'experiment1_' — prepended to output filenames


# ============================================================
#  RUN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if RUN_SOLVER:
        history = run_solver()
        # Auto-generate prefix from settings
        prefix = NAME_PREFIX or f"{SOLVER.lower()}_{CONTEXT}_"
        save_history(history, os.path.join(OUTPUT_DIR, f'{prefix}history.npz'))
    else:
        print(f"Loading history from {HISTORY_FILE}")
        history = load_history(HISTORY_FILE)
        prefix = NAME_PREFIX or os.path.splitext(os.path.basename(HISTORY_FILE))[0] + '_'

    print(f"\nGenerating plots (window={WINDOW}, bins={N_BINS}, topk={TOP_K})...")

    plot_family_probs(
        history, window=WINDOW,
        output_path=os.path.join(OUTPUT_DIR, f'{prefix}family.png'),
    )
    plot_action_heatmap(
        history, n_bins=N_BINS,
        output_path=os.path.join(OUTPUT_DIR, f'{prefix}heatmap.png'),
    )
    plot_topk_trajectories(
        history, k=TOP_K, window=WINDOW,
        output_path=os.path.join(OUTPUT_DIR, f'{prefix}topk.png'),
    )

    print("Done!")


def run_solver():
    if CONTEXT == 'benchmark' and SOLVER == 'DRLH':
        from Benchmark.drl.solve import solve_with_sb3
        print(f"Running Benchmark DRLH on {os.path.basename(INSTANCE)} (deterministic={DETERMINISTIC})")
        _, _, _, history = solve_with_sb3(INSTANCE, MODEL, deterministic=DETERMINISTIC)
        return history

    elif CONTEXT == 'benchmark' and SOLVER == 'RRT':
        from Benchmark.alns.alns_RRT import run_alns
        print(f"Running Benchmark RRT on {os.path.basename(INSTANCE)}")
        _, _, history = run_alns(INSTANCE)
        return history

    elif CONTEXT == 'case_study' and SOLVER == 'DRLH':
        from Case_study.alns.solve_drlh import solve_case_study
        print(f"Running Case Study DRLH (day={DELIVERY_DAY})")
        _, history = solve_case_study(
            customers_file=CUSTOMERS_FILE, delivery_day=DELIVERY_DAY,
        )
        return history

    elif CONTEXT == 'case_study' and SOLVER == 'RRT':
        from Case_study.alns.alns import run_alns
        print(f"Running Case Study RRT (day={DELIVERY_DAY})")
        _, history = run_alns(
            delivery_day=DELIVERY_DAY, customers_file=CUSTOMERS_FILE,
        )
        return history

    else:
        raise ValueError(f"Unknown combo: CONTEXT={CONTEXT}, SOLVER={SOLVER}")


if __name__ == '__main__':
    main()
