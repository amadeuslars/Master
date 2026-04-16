"""
Run ALNS on all training instances and delete infeasible ones (any unassigned).
Uses fewer iterations (3000) for speed — just checking feasibility, not optimality.
"""
import sys
import os
import glob
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from Case_study.alns.alns import run_alns

INSTANCES_DIR = os.path.join(os.path.dirname(__file__), 'training_instances')
MAX_ITERATIONS = 3000  # Quick feasibility check


def main():
    # Temporarily override iteration count for speed
    import Case_study.alns.alns as alns_module
    original_iters = alns_module.MAX_ITERATIONS
    alns_module.MAX_ITERATIONS = MAX_ITERATIONS

    files = sorted(glob.glob(os.path.join(INSTANCES_DIR, 'instance_*.csv')))
    print(f"Validating {len(files)} instances with {MAX_ITERATIONS} iterations each...\n")

    feasible = []
    infeasible = []
    t_start = time.time()

    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        try:
            sol, _ = run_alns(customers_file=fpath)
            n_unassigned = len(sol.routes[-1])
            n_total = sum(len(r) for r in sol.routes)

            if n_unassigned == 0:
                feasible.append(fname)
                status = "OK"
            else:
                infeasible.append(fname)
                os.remove(fpath)
                status = f"DELETED ({n_unassigned} unassigned)"

            elapsed = time.time() - t_start
            avg = elapsed / (i + 1)
            remaining = avg * (len(files) - i - 1)
            print(f"[{i+1:3d}/{len(files)}] {fname}: {status} | "
                  f"ETA: {remaining/60:.1f}min")

        except Exception as e:
            infeasible.append(fname)
            os.remove(fpath)
            print(f"[{i+1:3d}/{len(files)}] {fname}: ERROR ({e}) — DELETED")

    # Restore
    alns_module.MAX_ITERATIONS = original_iters

    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"Done in {elapsed/60:.1f} min")
    print(f"Feasible: {len(feasible)}/{len(files)}")
    print(f"Infeasible (deleted): {len(infeasible)}/{len(files)}")
    if infeasible:
        print(f"Deleted: {infeasible}")


if __name__ == '__main__':
    main()
