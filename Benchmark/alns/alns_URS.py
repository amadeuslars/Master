import random
import sys
import os
import csv
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import (
    precompute_nearest_neighbors, 
    create_initial_solution, 
    evaluate_solution,
    load_raw_solomon_data,
    two_opt_local_search,
    cross_route_segment_relocation,
    simple_relocate)

from utils.ml import QLearningAgent
from utils.visualization import ALNSTracker
from utils.operators2 import (
    random_removal, 
    shaw_removal, 
    worst_removal, 
    cluster_removal,
    least_used_vehicle_removal,
    greedy_insertion,
    regret_insertion)

# --- Configuration ---
MAX_ITERATIONS = 1000
SEGMENT_SIZE = 50 

# RRT Parameters
RRT_START_PERCENTAGE = 0.10  # Deviation allowed at start (20%)


# ---------------------------------------------------------
#  MAIN ALNS LOOP (RRT)
# ---------------------------------------------------------

def run_alns(instance_file):

    # Load data from Solomon .txt instance
    customers_df, vehicles_df, dist_matrix, cust_addr_idx, cust_arrays = load_raw_solomon_data(instance_file)
    dist_matrix = dist_matrix.astype(np.float64)
    cust_addr_idx = cust_addr_idx.astype(np.intp)
    cust_arrays['demand'] = cust_arrays['demand'].astype(np.float64)
    cust_arrays['tw_start'] = cust_arrays['tw_start'].astype(np.float64)
    cust_arrays['tw_end'] = cust_arrays['tw_end'].astype(np.float64)
    cust_arrays['service_time'] = cust_arrays['service_time'].astype(np.float64)

    destroy_ops = [random_removal, worst_removal, cluster_removal, shaw_removal, least_used_vehicle_removal]
    repair_ops = [greedy_insertion, regret_insertion]

    num_customers = len(customers_df['customer_id'])
    num_real_vehicles = int(vehicles_df['num_vehicles'])
    neighbor_sets = None

    
    current_sol = create_initial_solution(num_customers, num_real_vehicles)
    evaluate_solution(current_sol, dist_matrix, cust_addr_idx)
    
    best_sol = current_sol.copy()
    best_sol._cost = current_sol._cost
    best_found_at = 0

    print(f"Initial Cost: {current_sol._cost:.2f}")
    print(f"Starting Main Loop (RRT Strategy). Start Deviation: {RRT_START_PERCENTAGE*100}%")
    
    for it in range(MAX_ITERATIONS):

        # Select operators using roulette wheel (weighted random)
        d_idx = np.random.choice(len(destroy_ops))
        r_idx = np.random.choice(len(repair_ops))

        # RRT Threshold
        remaining_ratio = (MAX_ITERATIONS - it) / MAX_ITERATIONS
        threshold_value = RRT_START_PERCENTAGE * remaining_ratio * best_sol._cost
        acceptance_threshold = best_sol._cost + threshold_value
        
        low = int(num_customers * 0.02)
        high = int(num_customers * 0.40)
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
        
        if new_cost < best_sol._cost:
            accepted = True
            new_global_best = True
            best_sol = repaired.copy()
            best_sol._cost = new_cost
            best_found_at = it
            print(f"Iter {it} [New Best]: {new_cost:.2f} (Vehicles: {sum(1 for r in best_sol.routes[:-1] if r)})")
            
        elif new_cost < current_cost:
            accepted = True
            
            
        elif new_cost < acceptance_threshold:
            accepted = True
            
            
        if accepted:
            current_sol = repaired
            
        if (it + 1) % SEGMENT_SIZE == 0:
            print(f"--- Iter {it+1} | Threshold: +{threshold_value:.2f} | Best: {best_sol._cost:.2f} | Cur: {current_sol._cost:.2f} ---")

    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(f"Best Cost: {best_sol._cost:.2f}")
    print(f"Best found at iteration: {best_found_at}")

    print("Routes:")
    for i, r in enumerate(best_sol.routes[:-1]):
        if r:
            load = sum(cust_arrays['demand'][c-1] for c in r)
            print(f"V{i+1}: {r} | Load: {load}")

    return best_sol._cost, best_found_at

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    NUM_RUNS = 10
    ALGORITHM = "URS"

    # --- Add instance paths here ---
    INSTANCES = [
        os.path.join(project_root, 'Benchmark', 'data', 'homberger_600', 'C1_6_1.TXT'),
        os.path.join(project_root, 'Benchmark', 'data', 'homberger_600', 'R1_6_1.TXT'),
        os.path.join(project_root, 'Benchmark', 'data', 'homberger_600', 'RC1_6_1.TXT'),
    ]

    if not INSTANCES:
        print("No instances specified. Add paths to the INSTANCES list.")
        sys.exit(1)

    output_csv = os.path.join(project_root, 'logs', 'urs_results.csv')
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['algorithm', 'instance', 'run', 'best_cost', 'iteration_found_best'])

        for instance_path in INSTANCES:
            instance_name = os.path.basename(instance_path)
            print(f"\n{'='*60}")
            print(f"INSTANCE: {instance_name}")
            print(f"{'='*60}")

            for run in range(1, NUM_RUNS + 1):
                print(f"\n--- Run {run}/{NUM_RUNS} ---")
                start = time.perf_counter()
                cost, best_iter = run_alns(instance_path)
                elapsed = time.perf_counter() - start
                print(f"Elapsed time: {elapsed:.3f}s")
                writer.writerow([ALGORITHM, instance_name, run, f"{cost:.2f}", best_iter])
                f.flush()

    print(f"\nResults saved to: {output_csv}")