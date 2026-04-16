"""
Generate 1000 synthetic instances and run ALNS on each.
Instances with unassigned customers are logged to infeasible_instances.csv.

Run from repo root: python Case_study/run_instances.py
"""
import sys
import os
import time
import csv
import io
from contextlib import redirect_stdout

# Ensure we run from repo root (load_vrp_data uses relative paths like Case_study/data/...)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
CASE_STUDY = os.path.join(REPO_ROOT, 'Case_study')
sys.path.insert(0, CASE_STUDY)
sys.path.insert(0, os.path.join(CASE_STUDY, 'alns'))

from data.generate_instances import generate_instances
from alns import run_alns

# --- Configuration ---
INSTANCES_DIR = 'Case_study/data/training_instances'
NUM_INSTANCES = 1000
ALNS_ITERATIONS = 20000
SEED = 42
INFEASIBLE_LOG = 'Case_study/data/infeasible_instances.csv'
MASTER_CSV = 'Case_study/data/customers.csv'
# ----------------------


def main():
    # Step 1: Generate instances (skip if all already exist)
    first = os.path.join(INSTANCES_DIR, 'instance_0001.csv')
    last = os.path.join(INSTANCES_DIR, f'instance_{NUM_INSTANCES:04d}.csv')
    if os.path.exists(first) and os.path.exists(last):
        print(f"Instances already exist in {INSTANCES_DIR}/, skipping generation.")
    else:
        print("=" * 60)
        print(f"Step 1: Generating {NUM_INSTANCES} instances")
        print("=" * 60)
        generate_instances(MASTER_CSV, INSTANCES_DIR, NUM_INSTANCES, SEED)

    # Step 2: Run ALNS on each instance
    print("\n" + "=" * 60)
    print(f"Step 2: Running ALNS ({ALNS_ITERATIONS} iter) on {NUM_INSTANCES} instances")
    print("=" * 60)

    infeasible = []

    for i in range(1, NUM_INSTANCES + 1):
        instance_file = os.path.join(INSTANCES_DIR, f'instance_{i:04d}.csv')
        if not os.path.exists(instance_file):
            print(f"  [SKIP] {instance_file} not found")
            continue

        t0 = time.time()
        try:
            # Suppress ALNS verbose output
            with redirect_stdout(io.StringIO()):
                sol, history = run_alns(
                    customers_file=instance_file,
                    max_iterations=ALNS_ITERATIONS,
                )

            dummy_route = sol.routes[-1]
            num_unassigned = len(dummy_route)
            num_customers = sum(len(r) for r in sol.routes)
            elapsed = time.time() - t0

            if num_unassigned > 0:
                infeasible.append({
                    'instance': f'instance_{i:04d}.csv',
                    'num_customers': num_customers,
                    'num_unassigned': num_unassigned,
                    'best_cost': f'{sol.cost:.2f}',
                    'time_s': f'{elapsed:.1f}',
                })
                print(f"  [{i:4d}/{NUM_INSTANCES}] UNASSIGNED: {num_unassigned} customers "
                      f"(cost={sol.cost:.2f}, {elapsed:.1f}s)")
            else:
                print(f"  [{i:4d}/{NUM_INSTANCES}] OK "
                      f"(cost={sol.cost:.2f}, {elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            infeasible.append({
                'instance': f'instance_{i:04d}.csv',
                'num_customers': 0,
                'num_unassigned': -1,
                'best_cost': 'ERROR',
                'time_s': f'{elapsed:.1f}',
            })
            print(f"  [{i:4d}/{NUM_INSTANCES}] ERROR: {e}")

    # Step 3: Write infeasible log
    print("\n" + "=" * 60)
    print(f"Done. {len(infeasible)}/{NUM_INSTANCES} instances had unassigned customers.")
    print("=" * 60)

    if infeasible:
        with open(INFEASIBLE_LOG, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['instance', 'num_customers', 'num_unassigned', 'best_cost', 'time_s'])
            writer.writeheader()
            writer.writerows(infeasible)
        print(f"Infeasible instances logged to {INFEASIBLE_LOG}")
    else:
        print("All instances fully assigned!")


if __name__ == '__main__':
    main()
