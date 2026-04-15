import pandas as pd
import numpy as np
import os
import argparse


# --- Shift constants ---
LOADING_TIME = 1.0      # hours
DELOADING_TIME = 1/3    # 20 min in hours
S1_DEPART = 6.0 + LOADING_TIME   # 07:00
S1_END = 14.0
S2_DEPART = 15.0 + LOADING_TIME  # 16:00
S2_END = 23.0


# --- Demand generation by class ---

def generate_ppl(demand_class, rng):
    """Sample PPL for a single customer based on demand class."""
    if demand_class == 'low':
        val = rng.uniform(0.5, 1.0)
    elif demand_class == 'med':
        val = rng.uniform(1.0, 4.0)
    else:  # high
        val = max(2.0, rng.lognormal(mean=1.5, sigma=0.7))
    return round(val * 2) / 2  # round to nearest 0.5


def generate_ppl_freeze(ppl, rng):
    """~64% of deliveries have freeze cargo, proportional to PPL."""
    if rng.random() < 0.64:
        val = rng.uniform(0.25, min(ppl * 0.5, 4.0))
        return round(val * 2) / 2
    return 0.0


# --- Time window generation ---
# Slots and their approximate probabilities from full_week_data.csv
TW_START_SLOTS = [
    (6 * 60,  7 * 60,  0.04),   # 06:00-07:00
    (7 * 60,  9 * 60,  0.34),   # 07:00-09:00
    (9 * 60,  12 * 60, 0.35),   # 09:00-12:00
    (12 * 60, 15 * 60, 0.07),   # 12:00-15:00
    (15 * 60, 18 * 60, 0.15),   # 15:00-18:00
    (18 * 60, 21 * 60, 0.05),   # 18:00-21:00
]
TW_WINDOW_MINUTES = {1: 120, 2: 240}  # bracket 1: 2h, bracket 2: 4h


def load_master_matrix(matrix_dir):
    """Load master time matrix and build coordinate lookup."""
    time_matrix_file = os.path.join(matrix_dir, 'time_matrix.csv')
    time_df = pd.read_csv(time_matrix_file, index_col=0)
    time_matrix = time_df.values / 3600.0  # seconds to hours

    # Build coord label -> index mapping
    col_to_idx = {col.strip(): i for i, col in enumerate(time_df.columns)}

    # Parse depot (index 0)
    depot_idx = 0

    return time_matrix, col_to_idx, depot_idx


def get_travel_times(lat, lon, time_matrix, col_to_idx, depot_idx):
    """Get travel time from depot to customer and back."""
    coord_key = f"{lat:.7f}_{lon:.7f}"
    idx = col_to_idx.get(coord_key)
    if idx is None:
        # Fuzzy match
        best_dist = float('inf')
        best_idx = depot_idx
        for col, cidx in col_to_idx.items():
            parts = col.split('_')
            if len(parts) == 2:
                try:
                    clat, clon = float(parts[0]), float(parts[1])
                    d = (clat - lat)**2 + (clon - lon)**2
                    if d < best_dist:
                        best_dist = d
                        best_idx = cidx
                except ValueError:
                    pass
        idx = best_idx

    t_from_depot = time_matrix[depot_idx, idx]
    t_to_depot = time_matrix[idx, depot_idx]
    return t_from_depot, t_to_depot


def generate_time_window(rng, tw_bracket, t_from_depot, t_to_depot):
    """Generate a feasible time window respecting shift constraints."""
    window = TW_WINDOW_MINUTES[tw_bracket]
    window_h = window / 60.0

    # Compute feasible TW bounds for each shift
    # Earliest arrival: shift_depart + travel_from_depot
    # Latest tw_end: shift_end - deloading - travel_to_depot
    # tw_start must be <= tw_end - window_h (so the window fits)

    feasible_ranges = []  # list of (earliest_tw_start_mins, latest_tw_start_mins)

    for s_depart, s_end in [(S1_DEPART, S1_END), (S2_DEPART, S2_END)]:
        earliest_arrival_h = s_depart + t_from_depot
        latest_tw_end_h = s_end - DELOADING_TIME - t_to_depot

        # tw_start >= earliest_arrival - window_h (customer can arrive at tw_start, we wait)
        # But tw_start must be <= latest_tw_end - window_h so tw_end fits
        # Also tw_start >= earliest_arrival (vehicle can't arrive before tw_start...
        #   actually vehicle CAN arrive before tw_start and wait, so tw_start just needs
        #   to be reachable: earliest_arrival <= tw_end)
        # Simplest: tw_end <= latest_tw_end, and earliest_arrival <= tw_end

        latest_tw_end_mins = int(latest_tw_end_h * 60)
        earliest_arrival_mins = int(np.ceil(earliest_arrival_h * 60))

        # tw_end must be >= earliest_arrival (so vehicle can reach in time)
        # tw_end = tw_start + window
        # So tw_start + window >= earliest_arrival => tw_start >= earliest_arrival - window
        min_tw_start = max(0, earliest_arrival_mins - window)

        # tw_end <= latest_tw_end => tw_start <= latest_tw_end - window
        max_tw_start = latest_tw_end_mins - window

        if min_tw_start <= max_tw_start:
            feasible_ranges.append((min_tw_start, max_tw_start))

    if not feasible_ranges:
        # Fallback: wide window centered on day
        start_mins = 7 * 60
        end_mins = start_mins + window
        sh, sm = divmod(start_mins, 60)
        eh, em = divmod(end_mins, 60)
        return f"{sh:02d}:{sm:02d}", f"{eh:02d}:{em:02d}"

    # Pick a random feasible range (prefer S1 with realistic distribution)
    probs = [s[2] for s in TW_START_SLOTS]
    probs = np.array(probs) / sum(probs)

    # Try up to 20 times to find a slot that overlaps a feasible range
    for _ in range(20):
        slot_idx = rng.choice(len(TW_START_SLOTS), p=probs)
        slot_start, slot_end, _ = TW_START_SLOTS[slot_idx]

        for fmin, fmax in feasible_ranges:
            clamped_start = max(slot_start, fmin)
            clamped_end = min(slot_end, fmax)
            if clamped_start <= clamped_end:
                start_mins = int(rng.integers(clamped_start, clamped_end + 1))
                end_mins = start_mins + window
                sh, sm = divmod(start_mins, 60)
                eh, em = divmod(end_mins, 60)
                return f"{sh:02d}:{sm:02d}", f"{eh:02d}:{em:02d}"

    # Last resort: pick from any feasible range
    fmin, fmax = feasible_ranges[0]
    start_mins = int(rng.integers(fmin, fmax + 1))
    end_mins = start_mins + window
    sh, sm = divmod(start_mins, 60)
    eh, em = divmod(end_mins, 60)
    return f"{sh:02d}:{sm:02d}", f"{eh:02d}:{em:02d}"


def generate_instance(df_master, rng, time_matrix, col_to_idx, depot_idx):
    """Generate a single synthetic VRPTW instance."""
    # 1. Sample 150-250 customers weighted by frekvens
    n_customers = rng.integers(150, 251)
    n_customers = min(n_customers, len(df_master))

    weights = df_master['frekvens'].values.astype(float)
    weights = weights / weights.sum()

    indices = rng.choice(len(df_master), size=n_customers, replace=False, p=weights)
    df_inst = df_master.iloc[indices].copy()

    # 2. Generate demands
    df_inst['ppl'] = [generate_ppl(dc, rng) for dc in df_inst['demand_class']]
    df_inst['ppl_freeze'] = [generate_ppl_freeze(p, rng) for p in df_inst['ppl']]
    df_inst['volume_m3'] = 0.0
    df_inst['weight_kg'] = 0.0

    # 3. Generate feasible time windows using travel times from master matrix
    tw_data = []
    for _, row in df_inst.iterrows():
        t_from, t_to = get_travel_times(
            row['latitude'], row['longitude'],
            time_matrix, col_to_idx, depot_idx
        )
        tw = generate_time_window(rng, tw_bracket=row['tw_bracket'], t_from_depot=t_from, t_to_depot=t_to)
        tw_data.append(tw)

    df_inst['tw_start'] = [tw[0] for tw in tw_data]
    df_inst['tw_end'] = [tw[1] for tw in tw_data]

    # 4. Service time variation +/- 15%
    noise = rng.uniform(0.85, 1.15, size=len(df_inst))
    df_inst['service_time_min'] = (df_inst['service_time_min'] * noise).clip(lower=1.0).round(1)

    # Output columns matching load_vrp_data format
    out_cols = [
        'customer_id', 'customer_name', 'address', 'postal_code',
        'latitude', 'longitude',
        'tw_start', 'tw_end', 'ppl', 'ppl_freeze',
        'volume_m3', 'weight_kg', 'vehicle_type',
        'tour_id', 'service_time_min'
    ]
    return df_inst[out_cols].reset_index(drop=True)


def generate_instances(input_csv, output_dir, num_instances=1000, seed=42):
    """Generate multiple synthetic VRPTW instances from master customer file."""
    print(f"Loading master customer data from {input_csv}")
    df_master = pd.read_csv(input_csv)
    print(f"  {len(df_master)} customers, frekvens range: {df_master['frekvens'].min()}-{df_master['frekvens'].max()}")

    # Load master matrix for travel time lookups
    matrix_dir = os.path.join(os.path.dirname(input_csv), 'matrices', 'master')
    print(f"  Loading master matrix from {matrix_dir}")
    time_matrix, col_to_idx, depot_idx = load_master_matrix(matrix_dir)

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    print(f"Generating {num_instances} instances in '{output_dir}/'")
    for i in range(num_instances):
        df_inst = generate_instance(df_master, rng, time_matrix, col_to_idx, depot_idx)
        output_path = os.path.join(output_dir, f"instance_{i+1:04d}.csv")
        df_inst.to_csv(output_path, index=False)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1} / {num_instances} done (last: {len(df_inst)} customers, {df_inst['ppl'].sum():.0f} total PPL)")

    print(f"Done. {num_instances} instances saved to '{output_dir}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic VRPTW instances")
    parser.add_argument("--input", default="customers.csv", help="Master customer CSV")
    parser.add_argument("--output", default="synthetic_instances", help="Output directory")
    parser.add_argument("--num", type=int, default=1000, help="Number of instances")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_instances(args.input, args.output, args.num, args.seed)
