import numpy as np
import random
import math
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
    regret_insertion)

# --- Configuration ---
DUMMY_VEHICLE_NAME = 'dummy'
DUMMY_PENALTY = 10000.0
MAX_ITERATIONS = 10000
SEGMENT_SIZE = 50 

# SA Parameters
START_TEMPERATURE = 200.0
WARMUP_ITERATIONS = 100   
ESCAPE_THRESHOLD = 1000

# Scoring (Rewards for RL)
SCORE_NEW_GLOBAL_BEST = 35
SCORE_BETTER_THAN_CURRENT = 10
SCORE_ACCEPTED_WORSE = 5
SCORE_REJECTED = 0


# ---------------------------------------------------------
#  MAIN ALNS LOOP
# ---------------------------------------------------------


def run_alns():
    customers_df, vehicles_df, _, dist_matrix, cust_addr_idx, cust_arrays = load_vrp_data()
    
    # Setup operators & Tracker
    # Added Shaw Removal to the list
    destroy_ops = [random_removal, worst_removal, cluster_removal, shaw_removal]
    repair_ops = [greedy_insertion, regret_insertion]
    
    destroy_names = ['Random', 'Worst', 'Cluster', 'Shaw']
    repair_names = ['Greedy', 'Regret']
    tracker = ALNSTracker(destroy_names, repair_names)

    num_customers = len(customers_df)
    try:
        num_real_vehicles = int(vehicles_df.loc['Standard', 'num_vehicles'])
    except KeyError:
        num_real_vehicles = 25 # Default higher for 200 nodes

    print("Precomputing nearest neighbors...")
    neighbor_sets = precompute_nearest_neighbors(dist_matrix, num_neighbors=10)
    
    agent = QLearningAgent(len(destroy_ops), len(repair_ops))
    
    # Initial Solution
    current_sol = create_initial_solution(num_customers, num_real_vehicles)
    evaluate_solution(current_sol, dist_matrix, cust_addr_idx)
    
    best_sol = current_sol.copy()
    best_sol._cost = current_sol._cost
    curr_temp = START_TEMPERATURE
    
    iter_no_improve = 0  # Counts iterations since last global best
    
    print(f"Initial Cost: {current_sol._cost:.2f}")
    
    # --- Warm-up ---
    print(f"Warming up ({WARMUP_ITERATIONS} iters)...")
    deltas = []
    for it in range(WARMUP_ITERATIONS):
        # Only use Random/Greedy for fast warmup
        d_idx = 0 
        r_idx = 0
        
        n_remove = random.randint(int(num_customers * 0.1), int(num_customers * 0.2))
        temp_sol = destroy_ops[d_idx](current_sol, n_remove, distance_matrix_array=dist_matrix, customer_addr_idx=cust_addr_idx, customer_arrays=cust_arrays)
        temp_sol = repair_ops[r_idx](temp_sol, distance_matrix_array=dist_matrix, customer_addr_idx=cust_addr_idx, customer_arrays=cust_arrays, vehicles_df=vehicles_df, neighbor_sets=neighbor_sets)
        
        c = evaluate_solution(temp_sol, dist_matrix, cust_addr_idx)
        deltas.append(abs(c - current_sol._cost))
        
        if c < current_sol._cost:
            current_sol = temp_sol
            if c < best_sol._cost:
                best_sol = temp_sol.copy()
                best_sol._cost = c

    # Temperature Calibration
    typical_delta = np.mean(deltas) if deltas else 50.0
    FINAL_TEMPERATURE = 0.5
    # Calculate cooling to reach final temp at MAX_ITERATIONS
    cooling_rate = (FINAL_TEMPERATURE / START_TEMPERATURE) ** (1.0 / (MAX_ITERATIONS - WARMUP_ITERATIONS))
    
    print(f"Starting Main Loop. Start Temp: {curr_temp:.1f}, Cooling: {cooling_rate:.5f}")
    
    # --- Main Loop ---
    escaped = False
    for it in range(WARMUP_ITERATIONS, MAX_ITERATIONS):
        # Escape Mechanism
        if iter_no_improve >= ESCAPE_THRESHOLD:
            escaped = True
            print(f"Iter {it}: ESCAPE triggered.")
            n_esc = int(num_customers * 0.5) # Remove 50%
            current_sol = random_removal(best_sol, n_esc) # Reset from best
            current_sol = greedy_insertion(
                current_sol, 
                distance_matrix_array=dist_matrix, 
                customer_addr_idx=cust_addr_idx, 
                customer_arrays=cust_arrays, 
                vehicles_df=vehicles_df, 
                neighbor_sets=neighbor_sets
            )
            evaluate_solution(current_sol, dist_matrix, cust_addr_idx)
            iter_no_improve = 0
            # curr_temp = START_TEMPERATURE * 0.5 # Reheat

        state = agent.get_state(it, MAX_ITERATIONS)
        d_idx = agent.select_action(state, agent.q_destroy)
        r_idx = agent.select_action(state, agent.q_repair)
        
        # Dynamic removal size (10% to 30% of customers)
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
        
        # --- Local Search ---
        # 1. Intra-Route (2-opt): Frequent
        if random.random() < 0.4: 
            two_opt_local_search(repaired, dist_matrix, cust_addr_idx, cust_arrays)

        # 2. Inter-Route (Relocate): Moderate frequency
        if random.random() < 0.1:
            simple_relocate(repaired, dist_matrix, cust_addr_idx, cust_arrays, vehicles_df)
        
        # 3. Heavy Segment Relocation: Periodic or if promising
        pre_eval = evaluate_solution(repaired, dist_matrix, cust_addr_idx)
        if (pre_eval < current_sol._cost) or (it % 250 == 0):
            cross_route_segment_relocation(repaired, dist_matrix, cust_addr_idx, cust_arrays, vehicles_df)
            evaluate_solution(repaired, dist_matrix, cust_addr_idx) # Re-eval after move
            
        new_cost = repaired._cost
        current_cost = current_sol._cost
        
        # Acceptance
        accepted = False
        new_global_best = False
        reward = SCORE_REJECTED
        delta = new_cost - current_cost
        
        if delta < 0:
            accepted = True
            if new_cost < best_sol._cost:
                new_global_best = True
                reward = SCORE_NEW_GLOBAL_BEST
                best_sol = repaired.copy()
                best_sol._cost = new_cost
                print(f"Iter {it} [New Best]: {new_cost:.2f} (Vehicles: {sum(1 for r in best_sol.routes[:-1] if r)})")
            else:
                reward = SCORE_BETTER_THAN_CURRENT
        else:
            if random.random() < math.exp(-delta / curr_temp):
                accepted = True
                reward = SCORE_ACCEPTED_WORSE
        
        if accepted:
            current_sol = repaired
        
        if escaped:
            iter_no_improve = 0
            escaped = False
        elif new_global_best:
            iter_no_improve = 0
        else:
            iter_no_improve += 1
        
        # RL Update
        next_s = agent.get_state(it + 1, MAX_ITERATIONS)
        agent.update(state, d_idx, reward, next_s, agent.q_destroy)
        agent.update(state, r_idx, reward, next_s, agent.q_repair)
        
        # Logging
        if it % 10 == 0:
            tracker.record_iteration(it, best_sol._cost, current_sol._cost,
                                     agent.q_destroy[state].tolist(),
                                     agent.q_repair[state].tolist(),
                                     (reward == SCORE_NEW_GLOBAL_BEST))
            
        if (it + 1) % SEGMENT_SIZE == 0:
            print(f"--- Iter {it+1} | Temp: {curr_temp:.2f} | Best: {best_sol._cost:.2f} | Current: {current_sol._cost:.2f} ---")


        curr_temp *= cooling_rate

    # Final Report
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(f"Best Cost: {best_sol._cost:.2f}")
    
    print("Routes:")
    for i, r in enumerate(best_sol.routes[:-1]):
        if r:
            load = sum(cust_arrays['demand'][c-1] for c in r)
            print(f"V{i+1}: {r} | Load: {load}")
            
    tracker.plot_all(prefix='alns_optimized', save=True, show=False)

if __name__ == "__main__":
    run_alns()