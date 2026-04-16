import numpy as np
import time
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import load_vrp_data, generate_initial_solution, evaluate_solution
try:
    from utils.operators_cy import (
        random_removal, worst_removal, cluster_removal, shaw_removal,
        greedy_insertion, regret_insertion,
        two_opt_local_search, or_opt_local_search,
        get_earliest_departure, LOADING_TIME, DELOADING_TIME,
    )
    _USING_CYTHON = True
except ImportError:
    from utils.operators import (
        random_removal, worst_removal, cluster_removal, shaw_removal,
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
MAX_ITERATIONS = 100000
SEGMENT_SIZE = 50

# RRT Parameters
RRT_START_PERCENTAGE = 0.05

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

def run_alns(delivery_day='tue', customers_file='Case_study/data/customers.csv', max_iterations=None):

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
    # Build 30 composite actions (3 destroy × 5 sizes × 2 repair)
    # Encoding: action_idx = d_idx * 10 + s_idx * 2 + r_idx
    destroy_ops = [
        ('random', random_removal),
        ('worst', worst_removal),
        ('cluster', cluster_removal),
        ('shaw', shaw_removal),
    ]
    repair_ops = [
        ('greedy', greedy_insertion),
        ('regret', regret_insertion),
    ]
    NUM_ACTIONS = len(destroy_ops) * len(REMOVAL_SIZES) * len(repair_ops)  # 40

    actions = []
    for d_name, d_op in destroy_ops:
        for s_name, lo, hi in REMOVAL_SIZES:
            for r_name, r_op in repair_ops:
                actions.append({
                    'd_name': d_name, 'd_op': d_op,
                    's_name': s_name, 'lo': lo, 'hi': hi,
                    'r_name': r_name, 'r_op': r_op,
                    'label': f"{d_name}_{s_name}_{r_name}",
                })

    # Initialize single composite roulette wheel (one weight per action)
    action_weights = np.ones(NUM_ACTIONS)

    # Generate initial solution: 72 route slots (18 vehicles x 4 trips) + 1 dummy
    current_sol = generate_initial_solution(customers_dict, vehicle_names=sorted_vehicle_names)
    evaluate_solution(current_sol, addr_idx, time_matrix_array, depot_idx)
    best_sol = current_sol.copy()
    best_sol.cost = current_sol.cost

    num_customers = len(customers_dict['customer_id'])
    num_slots = len(current_sol.routes) - 1  # exclude dummy
    print(f"Multi-trip ALNS: {num_customers} customers, {len(vehicle_names)} vehicles, {num_slots} route slots")
    print(f"Initial Cost: {current_sol.cost:.2f}")

    # History logging
    history = {
        'iterations': [],
        'actions': [],
        'costs': [],
        'action_weights': [],
        'algorithm': 'RRT',
    }

    num_iterations = max_iterations if max_iterations is not None else MAX_ITERATIONS
    for i in range(num_iterations):
        # Select composite action using roulette wheel
        action_probs = action_weights / action_weights.sum()
        action_idx = np.random.choice(NUM_ACTIONS, p=action_probs)
        act = actions[action_idx]

        # RRT Threshold
        remaining_ratio = (num_iterations - i) / num_iterations
        threshold_value = RRT_START_PERCENTAGE * remaining_ratio * best_sol.cost
        acceptance_threshold = best_sol.cost + threshold_value

        n_remove = random.randint(act['lo'], min(act['hi'], num_customers))

        destroyed = act['d_op'](current_sol, n_remove, time_matrix_array=time_matrix_array, customer_addr_idx=addr_idx, customer_arrays=customer_arrays, depot_idx=depot_idx)
        repaired = act['r_op'](
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

        # Update composite action weight
        action_weights[action_idx] = WEIGHT_DECAY * action_weights[action_idx] + (1 - WEIGHT_DECAY) * reward

        # Log history
        history['iterations'].append(i)
        history['actions'].append(int(action_idx))
        history['costs'].append(best_sol.cost)
        history['action_weights'].append(action_weights.copy())

        if (i + 1) % SEGMENT_SIZE == 0:
            print(f"--- Iter {i+1} | Threshold: +{threshold_value:.2f} | Best: {best_sol.cost:.2f} | Cur: {current_sol.cost:.2f} ---")

    # Convert to numpy arrays for efficient storage/plotting
    history['action_weights'] = np.array(history['action_weights'])
    history['actions'] = np.array(history['actions'])
    history['costs'] = np.array(history['costs'])
    history['action_labels'] = [a['label'] for a in actions]

    # --- Final Results ---
    _print_final_results(best_sol, customers_dict, num_customers,
                         time_matrix_array, addr_idx, customer_arrays, depot_idx)

    return best_sol, history


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

        label = f"{veh} T{meta['trip']}" if meta else veh
        print(f"\n  === {label} ({len(route)} stops) ===")
        for ev in events:
            cust_label = ""
            if ev['customer'] is not None:
                cust_label = f" [#{int(cids[ev['customer'] - 1])}]"
            print(f"    {_fmt_time(ev['time'])}  {ev['details']}{cust_label}")

        # Show end time
        if events:
            last_time = events[-1]['time']
            print(f"    -- Done at {_fmt_time(last_time)} --")


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
        label = f"{veh} T{meta['trip']}" if meta else veh
        print(f"  [{i:2d}] {label}: {route}")

    # Print full schedule if data is available
    if time_matrix_array is not None:
        print_schedule(best_sol, customers_dict, time_matrix_array,
                       customer_addr_idx, customer_arrays, depot_idx)

    if best_sol.routes[-1]:
        unassigned_kundenr = [int(cids[c - 1]) for c in best_sol.routes[-1]]
        print(f"\n  Unassigned ({num_unassigned}): {unassigned_kundenr}")


if __name__ == "__main__":
    sol, history = run_alns(customers_file='Case_study/data/training_instances/real_tue.csv')
