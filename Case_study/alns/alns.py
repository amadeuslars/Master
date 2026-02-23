import numpy as np
import time
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import load_vrp_data, generate_initial_solution, evaluate_solution
from utils.operators import (
    random_removal, worst_removal, cluster_removal, shaw_removal,
    least_used_vehicle_removal, greedy_insertion, regret_insertion
)
# --- Configuration ---
MAX_ITERATIONS = 1000
SEGMENT_SIZE = 50 

# RRT Parameters
RRT_START_PERCENTAGE = 0.10  # Deviation allowed at start (20%)

# Scoring (Rewards for RL)
SCORE_NEW_GLOBAL_BEST = 5
SCORE_BETTER_THAN_CURRENT = 3
SCORE_ACCEPTED_WORSE = 1
SCORE_REJECTED = 0

# Roulette Wheel Parameters
WEIGHT_DECAY = 0.8  # How much to decay old weights (0.8 = keep 80% of old weight)

# ---------------------------------------------------------
#  MAIN ALNS LOOP (RRT)
# ---------------------------------------------------------

def run_alns():
    
    customers_dict, vehicles_dict, vehicle_names, time_matrix_array, distance_matrix_array, depot_idx, addr_idx, customer_arrays = load_vrp_data()

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
    destroy_ops = [random_removal, worst_removal, cluster_removal, shaw_removal, least_used_vehicle_removal]
    repair_ops = [greedy_insertion, regret_insertion]

    destroy_names = ['Random', 'Worst', 'Cluster', 'Shaw', 'Least']
    repair_names = ['Greedy', 'Regret']

    # Initialize roulette wheel weights (start equal for all operators)
    destroy_weights = np.ones(len(destroy_ops))
    repair_weights = np.ones(len(repair_ops))

    # Generate initial dummy solution with correct vehicle names
    current_sol = generate_initial_solution(customers_dict, vehicle_names=sorted_vehicle_names)
    evaluate_solution(current_sol, addr_idx, time_matrix_array, depot_idx)
    best_sol = current_sol.copy()
    best_sol.cost = current_sol.cost


    print(f"Initial Cost: {current_sol.cost:.2f}")

    for i in range(MAX_ITERATIONS):
        d_probs = destroy_weights / destroy_weights.sum()
        r_probs = repair_weights / repair_weights.sum()
        d_idx = np.random.choice(len(destroy_names), p=d_probs)
        r_idx = np.random.choice(len(repair_names), p=r_probs)

        # RRT Threshold
        remaining_ratio = (MAX_ITERATIONS - i) / MAX_ITERATIONS
        threshold_value = RRT_START_PERCENTAGE * remaining_ratio * best_sol.cost
        acceptance_threshold = best_sol.cost + threshold_value

        n_remove = random.randint(int(len(customers_dict['customer_id']) * 0.02), int(len(customers_dict['customer_id']) * 0.4))     

        destroyed = destroy_ops[d_idx](current_sol, n_remove, distance_matrix_array=distance_matrix_array, customer_addr_idx=addr_idx, customer_arrays=customer_arrays, depot_idx=depot_idx)
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
            new_global_best = True
            reward = SCORE_NEW_GLOBAL_BEST
            best_sol = repaired.copy()
            best_sol.cost = new_cost
            print(f"Iter {i} [New Best]: {new_cost:.2f} (Vehicles: {sum(1 for r in best_sol.routes[:-1] if r)})")
            
        elif new_cost < current_cost:
            accepted = True
            reward = SCORE_BETTER_THAN_CURRENT
            
        elif new_cost < acceptance_threshold:
            accepted = True
            reward = SCORE_ACCEPTED_WORSE
            
        if accepted:
            current_sol = repaired


        # Update action weight based on performance (roulette wheel)
        destroy_weights[d_idx] = WEIGHT_DECAY * destroy_weights[d_idx] + (1 - WEIGHT_DECAY) * reward
        repair_weights[r_idx] = WEIGHT_DECAY * repair_weights[r_idx] + (1- WEIGHT_DECAY) * reward
    
        if (i + 1) % SEGMENT_SIZE == 0:
            print(f"--- Iter {i+1} | Threshold: +{threshold_value:.2f} | Best: {best_sol.cost:.2f} | Cur: {current_sol.cost:.2f} ---")
    
    print("\n--- Final Results ---")
    print(f"Best Cost: {best_sol.cost:.2f} hours")
    print(f"Vehicles used: {sum(1 for i, r in enumerate(best_sol.routes) if r and best_sol.vehicles[i] != 'dummy')}")
    print()
    cids = customers_dict['customer_id']
    for i, (route, veh) in enumerate(zip(best_sol.routes, best_sol.vehicles)):
        if route and veh != 'dummy':
            kunde =[c for c in route]
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
    

    