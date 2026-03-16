import numpy as np
import time
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import load_vrp_data, generate_initial_solution, evaluate_solution
try:
    from utils.operators_cy import (
        random_removal, worst_removal, cluster_removal,
        greedy_insertion, regret_insertion,
        two_opt_local_search, or_opt_local_search,
        get_earliest_departure, LOADING_TIME, DELOADING_TIME,
    )
    _USING_CYTHON = True
except ImportError:
    from utils.operators import (
        random_removal, worst_removal, cluster_removal,
        greedy_insertion, regret_insertion,
        two_opt_local_search, or_opt_local_search
    )
    from utils.feasibility import (
        get_earliest_departure, LOADING_TIME, DELOADING_TIME
    )
    _USING_CYTHON = False

from utils.feasibility import compute_route_schedule
print(f"[ALNS] Backend: {'Cython' if _USING_CYTHON else 'Python'}")

# --- Configuration ---
MAX_ITERATIONS = 10000
SEGMENT_SIZE = 50

# RRT Parameters
RRT_START_PERCENTAGE = 0.10

# Removal size buckets (fixed counts, matching DRLH action space)
REMOVAL_SIZES = [
    ('xs', 2, 5),
    ('sm', 5, 10),
    ('md', 10, 20),
    ('lg', 20, 30),
    ('xl', 30, 40),
]

# Scoring (matching DRLH reward tiers)
SCORE_NEW_GLOBAL_BEST = 5
SCORE_BETTER_THAN_CURRENT = 3
SCORE_ACCEPTED_WORSE = 1
SCORE_REJECTED = 0

# Roulette Wheel Parameters
WEIGHT_DECAY = 0.8

# ---------------------------------------------------------
#  MAIN ALNS LOOP (RRT)
# ---------------------------------------------------------

def run_alns(delivery_day='tue', customers_file='Case_study/data/customers.csv'):

    customers_dict, vehicles_dict, vehicle_names, time_matrix_array, _, depot_idx, addr_idx, customer_arrays = load_vrp_data(delivery_day=delivery_day, customers_file=customers_file)

    real_vehicle_indices = [i for i, name in enumerate(vehicle_names) if name != 'dummy']
    # Sort by PPL total (descending)
    real_vehicles_sorted = sorted(real_vehicle_indices, key=lambda i: vehicles_dict[vehicle_names[i]]['PPL total'], reverse=True)
    sorted_vehicle_names = [vehicle_names[i] for i in real_vehicles_sorted] + ['dummy']

    # Pre-compute compatible PPLs for Biltype 2
    compatible_ppls_set = set()
    for v_name in ['small', 'medium-small', 'medium']:
        if v_name in vehicles_dict:
            compatible_ppls_set.add(vehicles_dict[v_name]['PPL total'])

    # --------------------------------
    destroy_ops = [random_removal, worst_removal, cluster_removal]
    repair_ops = [greedy_insertion, regret_insertion]

    destroy_names = ['Random', 'Worst', 'Cluster']
    repair_names = ['Greedy', 'Regret']

    # Initialize roulette wheel weights (start equal for all operators + buckets)
    destroy_weights = np.ones(len(destroy_ops))
    repair_weights = np.ones(len(repair_ops))
    bucket_weights = np.ones(len(REMOVAL_SIZES))

    # Generate initial solution: 72 route slots (18 vehicles x 4 trips) + 1 dummy
    current_sol = generate_initial_solution(customers_dict, vehicle_names=sorted_vehicle_names)
    evaluate_solution(current_sol, addr_idx, time_matrix_array, depot_idx)
    best_sol = current_sol.copy()
    best_sol.cost = current_sol.cost

    num_customers = len(customers_dict['customer_id'])
    num_slots = len(current_sol.routes) - 1  # exclude dummy
    print(f"Multi-trip ALNS: {num_customers} customers, {len(vehicle_names)} vehicles, {num_slots} route slots")
    print(f"Initial Cost: {current_sol.cost:.2f}")

    for i in range(MAX_ITERATIONS):
        d_probs = destroy_weights / destroy_weights.sum()
        r_probs = repair_weights / repair_weights.sum()
        b_probs = bucket_weights / bucket_weights.sum()
        d_idx = np.random.choice(len(destroy_names), p=d_probs)
        r_idx = np.random.choice(len(repair_names), p=r_probs)
        b_idx = np.random.choice(len(REMOVAL_SIZES), p=b_probs)

        # RRT Threshold
        remaining_ratio = (MAX_ITERATIONS - i) / MAX_ITERATIONS
        threshold_value = RRT_START_PERCENTAGE * remaining_ratio * best_sol.cost
        acceptance_threshold = best_sol.cost + threshold_value

        _, lo, hi = REMOVAL_SIZES[b_idx]
        n_remove = random.randint(lo, min(hi, num_customers))

        destroyed = destroy_ops[d_idx](current_sol, n_remove, time_matrix_array=time_matrix_array, customer_addr_idx=addr_idx, customer_arrays=customer_arrays, depot_idx=depot_idx)
        repaired = repair_ops[r_idx](
            destroyed,
            time_matrix_array,
            addr_idx,
            customer_arrays,
            vehicles_dict,
            None,  # neighbor_sets placeholder
            depot_idx=depot_idx,
            temperature=1.0,
            compatible_ppls_set=compatible_ppls_set
        )
        new_cost = evaluate_solution(repaired, addr_idx, time_matrix_array, depot_idx)
        current_cost = current_sol.cost

        accepted = False
        reward = SCORE_REJECTED

        if new_cost < best_sol.cost:
            accepted = True
            reward = SCORE_NEW_GLOBAL_BEST
        elif new_cost < current_cost:
            accepted = True
            reward = SCORE_BETTER_THAN_CURRENT
        elif new_cost < acceptance_threshold:
            accepted = True
            reward = SCORE_ACCEPTED_WORSE

        if accepted:
            # Local search disabled for fair comparison with DRLH (no LS in benchmark training)
            # repaired = two_opt_local_search(repaired, time_matrix_array, addr_idx, customer_arrays, depot_idx=depot_idx)
            # evaluate_solution(repaired, addr_idx, time_matrix_array, depot_idx)
            # repaired = or_opt_local_search(repaired, time_matrix_array, addr_idx, customer_arrays, depot_idx=depot_idx)
            # new_cost = evaluate_solution(repaired, addr_idx, time_matrix_array, depot_idx)
            current_sol = repaired
            if new_cost < best_sol.cost:
                best_sol = repaired.copy()
                best_sol.cost = new_cost
                num_assigned = sum(len(r) for r, v in zip(best_sol.routes, best_sol.vehicles) if v != 'dummy')
                num_trips = sum(1 for r, v in zip(best_sol.routes, best_sol.vehicles) if r and v != 'dummy')
                print(f"Iter {i} [New Best]: {new_cost:.2f} | Assigned: {num_assigned}/{num_customers} | Trips: {num_trips}")


        # Update action weights based on performance (roulette wheel)
        destroy_weights[d_idx] = WEIGHT_DECAY * destroy_weights[d_idx] + (1 - WEIGHT_DECAY) * reward
        repair_weights[r_idx] = WEIGHT_DECAY * repair_weights[r_idx] + (1 - WEIGHT_DECAY) * reward
        bucket_weights[b_idx] = WEIGHT_DECAY * bucket_weights[b_idx] + (1 - WEIGHT_DECAY) * reward

        if (i + 1) % SEGMENT_SIZE == 0:
            print(f"--- Iter {i+1} | Threshold: +{threshold_value:.2f} | Best: {best_sol.cost:.2f} | Cur: {current_sol.cost:.2f} ---")

    # --- Final Results ---
    _print_final_results(best_sol, customers_dict, num_customers,
                         time_matrix_array, addr_idx, customer_arrays, depot_idx)

    return best_sol


def _fmt_time(h):
    """Format hours as HH:MM."""
    hh = int(h)
    mm = int((h - hh) * 60)
    return f"{hh:02d}:{mm:02d}"


def print_schedule(solution, customers_dict, time_matrix_array, customer_addr_idx,
                   customer_arrays, depot_idx):
    """Print a detailed timeline for every active trip in the solution."""
    cids = customers_dict['customer_id']

    for i, (route, veh) in enumerate(zip(solution.routes, solution.vehicles)):
        if not route or veh == 'dummy':
            continue

        meta = solution.route_meta[i]
        earliest_dep = get_earliest_departure(
            solution, i, time_matrix_array, customer_addr_idx,
            customer_arrays, depot_idx
        )
        lunch_pos = solution.lunch_breaks[i]
        events = compute_route_schedule(
            route, time_matrix_array, customer_addr_idx,
            customer_arrays, depot_idx, earliest_dep, lunch_pos
        )

        label = f"{veh} S{meta['shift']}T{meta['trip']}" if meta else veh
        shift_end = meta['shift_end'] if meta else 22.0
        print(f"\n  === {label} ({len(route)} stops, shift ends {_fmt_time(shift_end)}) ===")
        for ev in events:
            cust_label = ""
            if ev['customer'] is not None:
                cust_label = f" [#{int(cids[ev['customer'] - 1])}]"
            print(f"    {_fmt_time(ev['time'])}  {ev['details']}{cust_label}")

        # Show slack
        if events:
            last_time = events[-1]['time']
            slack = shift_end - last_time
            print(f"    -- Slack: {slack*60:.0f} min until shift end --")


def _print_final_results(best_sol, customers_dict, num_customers,
                         time_matrix_array=None, customer_addr_idx=None,
                         customer_arrays=None, depot_idx=0):
    """Print final solution details with vehicle/shift/trip breakdown."""
    cids = customers_dict['customer_id']
    num_assigned = sum(len(r) for r, v in zip(best_sol.routes, best_sol.vehicles) if v != 'dummy')
    num_unassigned = len(best_sol.routes[-1])

    # Count unique vehicles used
    used_vehicles = set()
    num_trips = 0
    for i, (route, veh) in enumerate(zip(best_sol.routes, best_sol.vehicles)):
        if route and veh != 'dummy':
            num_trips += 1
            meta = best_sol.route_meta[i]
            if meta:
                used_vehicles.add(meta['vehicle_idx'])

    print(f"\n--- Final Results ---")
    print(f"Best Cost: {best_sol.cost:.2f} hours ({best_sol.cost * 60:.0f} min)")
    print(f"Assigned: {num_assigned}/{num_customers} | Unassigned: {num_unassigned}")
    print(f"Vehicles used: {len(used_vehicles)} | Active trips: {num_trips}")

    # Raw solution structure
    print(f"\n--- Raw Routes ---")
    for i, (route, veh) in enumerate(zip(best_sol.routes, best_sol.vehicles)):
        if not route:
            continue
        meta = best_sol.route_meta[i] if i < len(best_sol.route_meta) else None
        label = f"{veh} S{meta['shift']}T{meta['trip']}" if meta else veh
        print(f"  [{i:2d}] {label}: {route}")

    # Print full schedule if data is available
    if time_matrix_array is not None:
        print_schedule(best_sol, customers_dict, time_matrix_array,
                       customer_addr_idx, customer_arrays, depot_idx)

    if best_sol.routes[-1]:
        unassigned_kundenr = [int(cids[c - 1]) for c in best_sol.routes[-1]]
        print(f"\n  Unassigned ({num_unassigned}): {unassigned_kundenr}")


if __name__ == "__main__":
    sol = run_alns(delivery_day='tue', customers_file='Case_study/data/customers_alesund_sula_tue.csv')

