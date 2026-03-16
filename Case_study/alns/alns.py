import numpy as np
import time
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import load_vrp_data, generate_initial_solution, evaluate_solution
from utils.operators import (
    random_removal, worst_removal, cluster_removal,
    greedy_insertion, regret_insertion
)

# --- Configuration ---
MAX_ITERATIONS = 10000
SEGMENT_SIZE = 50 

# RRT Parameters
RRT_START_PERCENTAGE = 0.10  # Deviation allowed at start (10%)

# Scoring (Rewards for RL)
SCORE_NEW_GLOBAL_BEST = 5
SCORE_BETTER_THAN_CURRENT = 3
SCORE_ACCEPTED_WORSE = 1
SCORE_REJECTED = 0

# Roulette Wheel Parameters
WEIGHT_DECAY = 0.8  # How much to decay old weights

# ---------------------------------------------------------
#  ACTION SPACE SETUP (Composite Wheel)
# ---------------------------------------------------------

REMOVAL_SIZES = [
    ('xs', 2, 5),
    ('sm', 5, 10),
    ('md', 10, 20),
    ('lg', 20, 30),
    ('xl', 30, 40),
]

DESTROY_OPS = [
    ('random', random_removal),
    ('worst', worst_removal),
    ('cluster', cluster_removal),
]

REPAIR_OPS = [
    ('greedy', greedy_insertion),
    ('regret', regret_insertion),
]

NUM_ACTIONS = len(DESTROY_OPS) * len(REMOVAL_SIZES) * len(REPAIR_OPS) # 30 actions

def build_actions():
    """
    Builds 30 composite actions combining Destroy, Size, and Repair.
    Returns list of (sized_destroy_fn, repair_fn, label) tuples.
    """
    actions = []
    for d_name, d_op in DESTROY_OPS:
        for s_name, lo, hi in REMOVAL_SIZES:
            for r_name, r_op in REPAIR_OPS:
                
                # We use default arguments (_lo, _hi, _op) to avoid Python's late-binding loop issue
                def sized_destroy(solution, _lo=lo, _hi=hi, _op=d_op, **kwargs):
                    n_remove = random.randint(_lo, _hi)
                    return _op(solution, n_remove, **kwargs)

                label = f"{d_name}_{s_name}_{r_name}"
                actions.append((sized_destroy, r_op, label))

    return actions


# ---------------------------------------------------------
#  MAIN ALNS LOOP (RRT)
# ---------------------------------------------------------

def run_alns():
    
    customers_dict, vehicles_dict, vehicle_names, time_matrix_array, _, depot_idx, addr_idx, customer_arrays = load_vrp_data()

    real_vehicle_indices = [i for i, name in enumerate(vehicle_names) if name != 'dummy']
    # Sort by PPL total (descending)
    real_vehicles_sorted = sorted(real_vehicle_indices, key=lambda i: vehicles_dict[vehicle_names[i]]['PPL total'], reverse=True)
    sorted_vehicle_names = [vehicle_names[i] for i in real_vehicles_sorted] + ['dummy']

    # Pre-compute compatible PPLs for Biltype 2
    compatible_ppls_set = set()
    for v_name in ['small', 'medium-small', 'medium']:
        if v_name in vehicles_dict:
            compatible_ppls_set.add(vehicles_dict[v_name]['PPL total'])

    # Build the composite actions and initialize the single roulette wheel
    actions = build_actions()
    action_weights = np.ones(NUM_ACTIONS)

    # Generate initial dummy solution with correct vehicle names
    current_sol = generate_initial_solution(customers_dict, vehicle_names=sorted_vehicle_names)
    evaluate_solution(current_sol, addr_idx, time_matrix_array, depot_idx)
    best_sol = current_sol.copy()
    best_sol.cost = current_sol.cost

    print(f"Initial Cost: {current_sol.cost:.2f}")

    for i in range(MAX_ITERATIONS):
        
        # Select composite action using roulette wheel
        action_probs = action_weights / action_weights.sum()
        action_idx = np.random.choice(NUM_ACTIONS, p=action_probs)
        d_op, r_op, label = actions[action_idx]

        # RRT Threshold
        remaining_ratio = (MAX_ITERATIONS - i) / MAX_ITERATIONS
        threshold_value = RRT_START_PERCENTAGE * remaining_ratio * best_sol.cost
        acceptance_threshold = best_sol.cost + threshold_value

        # 1. DESTROY (using the sized wrapper)
        destroyed = d_op(
            current_sol, 
            time_matrix_array=time_matrix_array, 
            customer_addr_idx=addr_idx, 
            customer_arrays=customer_arrays, 
            depot_idx=depot_idx
        )
        
        # 2. REPAIR
        repaired = r_op(
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

        # 3. ACCEPTANCE & SCORING
        accepted = False
        reward = SCORE_REJECTED

        if new_cost < best_sol.cost:
            accepted = True
            reward = SCORE_NEW_GLOBAL_BEST
            best_sol = repaired.copy()
            best_sol.cost = new_cost
            print(f"Iter {i} [New Best]: {new_cost:.2f} (Vehicles: {sum(1 for r in best_sol.routes[:-1] if r)}) | Synergy: {label}")
        elif new_cost < current_cost:
            accepted = True
            reward = SCORE_BETTER_THAN_CURRENT
        elif new_cost < acceptance_threshold:
            accepted = True
            reward = SCORE_ACCEPTED_WORSE

        if accepted:
            current_sol = repaired

        # Update action weight based on performance
        action_weights[action_idx] = WEIGHT_DECAY * action_weights[action_idx] + (1 - WEIGHT_DECAY) * reward
    
        if (i + 1) % SEGMENT_SIZE == 0:
            print(f"--- Iter {i+1} | Threshold: +{threshold_value:.2f} | Best: {best_sol.cost:.2f} | Cur: {current_sol.cost:.2f} ---")
    
    # ---------------------------------------------------------
    #  OUTPUT FORMATTING
    # ---------------------------------------------------------
    print("\n--- Final Results ---")
    print(f"Best Cost: {best_sol.cost:.2f} hours")
    print(f"Vehicles used: {sum(1 for i, r in enumerate(best_sol.routes) if r and best_sol.vehicles[i] != 'dummy')}")
    print()
    
    cids = customers_dict['customer_id']
    for i, (route, veh) in enumerate(zip(best_sol.routes, best_sol.vehicles)):
        if route and veh != 'dummy':
            kunde = [c for c in route]
            kundenr = [int(cids[c - 1]) for c in route]
            lunch_pos = best_sol.lunch_breaks[i]
            if lunch_pos is not None:
                lunch_after = int(cids[route[lunch_pos - 1] - 1])
                print(f"  {veh}: {kundenr}  | Lunch after customer {lunch_after}")
                print(f'{veh}: {kunde}')
            else:
                print(f"  {veh}: {kundenr}  | No lunch scheduled")
                
    if best_sol.routes[-1]:
        unassigned_kundenr = [int(cids[c - 1]) for c in best_sol.routes[-1]]
        print(f"\n  Unassigned ({len(best_sol.routes[-1])}): {unassigned_kundenr}")

    return best_sol

if __name__ == "__main__":
    sol = run_alns()