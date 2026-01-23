import random
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import (
    precompute_nearest_neighbors, 
    create_initial_solution, 
    evaluate_solution, 
    load_vrp_data,
    two_opt_local_search,
    cross_route_segment_relocation,
    simple_relocate)

from utils.ml import QLearningAgent
from utils.visualization import ALNSTracker
from utils.operators import (
    random_removal, 
    shaw_removal, 
    worst_removal, 
    cluster_removal,
    greedy_insertion,
    regret_insertion,
    least_used_vehicle_removal)

# --- Configuration ---
DUMMY_VEHICLE_NAME = 'dummy'
DUMMY_PENALTY = 10000.0
MAX_ITERATIONS = 10000
SEGMENT_SIZE = 50 

# RRT Parameters
WARMUP_ITERATIONS = 100   
ESCAPE_THRESHOLD = 1000
RRT_START_PERCENTAGE = 0.10  # Deviation allowed at start (20%)

# Scoring (Rewards for RL)
SCORE_NEW_GLOBAL_BEST = 35
SCORE_BETTER_THAN_CURRENT = 10
SCORE_ACCEPTED_WORSE = 5
SCORE_REJECTED = 0

# Roulette Wheel Parameters
WEIGHT_DECAY = 0.8  # How much to decay old weights (0.8 = keep 80% of old weight)

# ---------------------------------------------------------
#  MAIN ALNS LOOP (RRT)
# ---------------------------------------------------------

def run_alns():
    customers_df, vehicles_df, _, dist_matrix, cust_addr_idx, cust_arrays = load_vrp_data()
    
    destroy_ops = [random_removal, worst_removal, cluster_removal, shaw_removal, least_used_vehicle_removal]
    repair_ops = [greedy_insertion, regret_insertion]
    
    destroy_names = ['Random', 'Worst', 'Cluster', 'Shaw', 'LeastUsed']
    repair_names = ['Greedy', 'Regret']
    tracker = ALNSTracker(destroy_names, repair_names)

    num_customers = len(customers_df)
    num_real_vehicles = int(vehicles_df.loc['Standard', 'num_vehicles'])
   
    print("Precomputing nearest neighbors...")
    neighbor_sets = precompute_nearest_neighbors(dist_matrix, num_neighbors=10)
    
    # Initialize roulette wheel weights (start equal for all operators)
    destroy_weights = np.ones(len(destroy_ops))
    repair_weights = np.ones(len(repair_ops))
    
    current_sol = create_initial_solution(num_customers, num_real_vehicles)
    evaluate_solution(current_sol, dist_matrix, cust_addr_idx)
    
    best_sol = current_sol.copy()
    best_sol._cost = current_sol._cost
    
    print(f"Initial Cost: {current_sol._cost:.2f}")
    print(f"Starting Main Loop (RRT Strategy). Start Deviation: {RRT_START_PERCENTAGE*100}%")
    
    for it in range(MAX_ITERATIONS):

        # Select operators using roulette wheel (weighted random)
        d_probs = destroy_weights / destroy_weights.sum()
        r_probs = repair_weights / repair_weights.sum()
        d_idx = np.random.choice(len(destroy_ops), p=d_probs)
        r_idx = np.random.choice(len(repair_ops), p=r_probs)

        # RRT Threshold
        remaining_ratio = (MAX_ITERATIONS - it) / MAX_ITERATIONS
        threshold_value = RRT_START_PERCENTAGE * remaining_ratio * best_sol._cost
        acceptance_threshold = best_sol._cost + threshold_value
        
        low = int(num_customers * 0.10)
        high = int(num_customers * 0.30)
        n_remove = random.randint(low, high)
        
        destroyed = destroy_ops[d_idx](
            current_sol, n_remove, 
            distance_matrix_array=dist_matrix, 
            customer_addr_idx=cust_addr_idx,
            customer_arrays=cust_arrays
        )
        
        repaired = repair_ops[r_idx](
            destroyed, 
            distance_matrix_array=dist_matrix, 
            customer_addr_idx=cust_addr_idx,
            customer_arrays=cust_arrays, 
            vehicles_df=vehicles_df, 
            neighbor_sets=neighbor_sets
        )
           
        new_cost = evaluate_solution(repaired, dist_matrix, cust_addr_idx)
        current_cost = current_sol._cost
        
        accepted = False
        reward = SCORE_REJECTED
        
        if new_cost < best_sol._cost:
            accepted = True
            new_global_best = True
            reward = SCORE_NEW_GLOBAL_BEST
            best_sol = repaired.copy()
            best_sol._cost = new_cost
            print(f"Iter {it} [New Best]: {new_cost:.2f} (Vehicles: {sum(1 for r in best_sol.routes[:-1] if r)})")
            
        elif new_cost < current_cost:
            accepted = True
            reward = SCORE_BETTER_THAN_CURRENT
            
        elif new_cost < acceptance_threshold:
            accepted = True
            reward = SCORE_ACCEPTED_WORSE
            
        if accepted:
            current_sol = repaired

        # Update operator weights based on performance (roulette wheel)
        destroy_weights[d_idx] = WEIGHT_DECAY * destroy_weights[d_idx] + (1 - WEIGHT_DECAY) * reward
        repair_weights[r_idx] = WEIGHT_DECAY * repair_weights[r_idx] + (1 - WEIGHT_DECAY) * reward
        
        # Track progress
        if it % 10 == 0:
            tracker.record_iteration(it, best_sol._cost, current_sol._cost,
                                     destroy_weights.tolist(),
                                     repair_weights)
            
        if (it + 1) % SEGMENT_SIZE == 0:
            print(f"--- Iter {it+1} | Threshold: +{threshold_value:.2f} | Best: {best_sol._cost:.2f} | Cur: {current_sol._cost:.2f} ---")

    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(f"Best Cost: {best_sol._cost:.2f}")
    
    print("Routes:")
    for i, r in enumerate(best_sol.routes[:-1]):
        if r:
            load = sum(cust_arrays['demand'][c-1] for c in r)
            print(f"V{i+1}: {r} | Load: {load}")
            
    tracker.plot_all(prefix='alns_rrt_optimized', save=False, show=False)

if __name__ == "__main__":
    run_alns()