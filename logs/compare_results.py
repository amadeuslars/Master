"""
Compare DRLH results across multiple CSV files.
Usage: python compare_results.py file1.csv file2.csv ...
       python compare_results.py  (uses hardcoded FILES list below)
"""
import sys
import os
import csv
from collections import defaultdict

# --- Configure files to compare here ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "rrt": os.path.join(SCRIPT_DIR, "drlh_results_homberger_600_10000iter_5310_all_stochastic.csv"),
    "DRLH":   os.path.join(SCRIPT_DIR, "drlh_results_homberger_600_10000iter_5310_all.csv")
}
# ----------------------------------------


def load_csv(path):
    """Returns {instance: [costs]} from a results CSV."""
    data = defaultdict(list)
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row['instance']].append(float(row['best_cost']))
    return data


def summarise(costs):
    n = len(costs)
    mean = sum(costs) / n
    best = min(costs)
    worst = max(costs)
    return mean, best, worst, n


def instance_type(name):
    """Extract type prefix: C1, C2, R1, R2, RC1, RC2."""
    upper = name.upper()
    for prefix in ('RC2', 'RC1', 'R2', 'R1', 'C2', 'C1'):
        if upper.startswith(prefix):
            return prefix
    return 'OTHER'


def main():
    # Allow overriding files via CLI args
    if len(sys.argv) > 1:
        files = {}
        for path in sys.argv[1:]:
            label = os.path.basename(path).replace('.csv', '')
            files[label] = path
    else:
        files = FILES

    # Load all datasets
    datasets = {}
    for label, path in files.items():
        if not os.path.exists(path):
            print(f"  [SKIP] {label}: file not found ({path})")
            continue
        datasets[label] = load_csv(path)
        print(f"  [OK]   {label}: {path}")

    if not datasets:
        print("No files loaded.")
        return

    # Collect all instances across all files
    all_instances = sorted(
        set(inst for d in datasets.values() for inst in d),
        key=lambda x: (instance_type(x), x)
    )

    labels = list(datasets.keys())
    col_w = 22  # label column width
    cost_w = 16

    # --- Per-instance table ---
    print("\n" + "=" * (24 + cost_w * len(labels) * 3))
    print("PER-INSTANCE AVERAGES (mean / best / worst over runs)")
    print("=" * (24 + cost_w * len(labels) * 3))

    # Header
    header = f"{'Instance':<22}"
    for lbl in labels:
        header += f"  {lbl[:cost_w*3-2]:^{cost_w*3-2}}"
    print(header)
    sub = " " * 22
    for _ in labels:
        sub += f"  {'mean':>{cost_w-2}}  {'best':>{cost_w-2}}  {'worst':>{cost_w-2}}"
    print(sub)
    print("-" * len(header))

    # Group by type for readability
    current_type = None
    type_sums = defaultdict(lambda: defaultdict(list))  # type -> label -> [means]

    for inst in all_instances:
        t = instance_type(inst)
        if t != current_type:
            if current_type is not None:
                print()
            current_type = t

        row = f"{inst:<22}"
        for lbl in labels:
            d = datasets[lbl]
            if inst in d:
                mean, best, worst, _ = summarise(d[inst])
                row += f"  {mean:>{cost_w-2}.2f}  {best:>{cost_w-2}.2f}  {worst:>{cost_w-2}.2f}"
                type_sums[t][lbl].append(mean)
            else:
                row += f"  {'—':>{cost_w-2}}  {'—':>{cost_w-2}}  {'—':>{cost_w-2}}"
        print(row)

    # --- Type-group summary ---
    print("\n" + "=" * (24 + cost_w * len(labels)))
    print("GROUP AVERAGES (mean-of-means per instance type)")
    print("=" * (24 + cost_w * len(labels)))
    header2 = f"{'Type':<22}"
    for lbl in labels:
        header2 += f"  {lbl[:cost_w-2]:>{cost_w-2}}"
    print(header2)
    print("-" * len(header2))

    all_type_means = defaultdict(list)  # label -> [group means]
    for t in sorted(type_sums.keys()):
        row = f"{t:<22}"
        for lbl in labels:
            vals = type_sums[t].get(lbl, [])
            if vals:
                gm = sum(vals) / len(vals)
                row += f"  {gm:>{cost_w-2}.2f}"
                all_type_means[lbl].append(gm)
            else:
                row += f"  {'—':>{cost_w-2}}"
        print(row)

    # Overall mean
    print("-" * len(header2))
    row = f"{'OVERALL':<22}"
    for lbl in labels:
        vals = all_type_means[lbl]
        if vals:
            row += f"  {sum(vals)/len(vals):>{cost_w-2}.2f}"
        else:
            row += f"  {'—':>{cost_w-2}}"
    print(row)

    # --- Delta column if exactly 2 files ---
    if len(labels) == 2:
        lbl_a, lbl_b = labels
        print(f"\n  Delta = {lbl_b} - {lbl_a}  (negative = improvement)")
        print("-" * 50)
        for t in sorted(type_sums.keys()):
            a_vals = type_sums[t].get(lbl_a, [])
            b_vals = type_sums[t].get(lbl_b, [])
            if a_vals and b_vals:
                delta = sum(b_vals)/len(b_vals) - sum(a_vals)/len(a_vals)
                pct = delta / (sum(a_vals)/len(a_vals)) * 100
                sign = "↓" if delta < 0 else "↑"
                print(f"  {t:<8}  {delta:+.2f}  ({pct:+.2f}%)  {sign}")


if __name__ == "__main__":
    main()
