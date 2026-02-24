import random
import sys
import os
import csv
import numpy as np
import time
import glob
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import ( 
    create_initial_solution, 
    evaluate_solution,
    load_raw_solomon_data,
    two_opt_local_search,
    cross_route_segment_relocation,
    simple_relocate)

from utils.ml import QLearningAgent
from utils.visualization import ALNSTracker
from utils.actions import build_actions, NUM_ACTIONS

# --- Configuration ---
MAX_ITERATIONS = 1000
SEGMENT_SIZE = 50

SA_START_TEMP = 500.0  # You will need to tune this based on your average move cost
SA_COOLING_RATE = 0.9995  # How quickly the temperature drops (e.g., 0.999 means slow decay)
# Simulated Annealing Parameters
SA_TARGET_PROB = 0.5  # 50% chance to accept a worse move at the start
SA_END_TEMP = 0.1

# Scoring (Rewards for RL)
SCORE_NEW_GLOBAL_BEST = 5
SCORE_BETTER_THAN_CURRENT = 3
SCORE_ACCEPTED_WORSE = 1
SCORE_REJECTED = 0

# Roulette Wheel Parameters
WEIGHT_DECAY = 0.8  # How much to decay old weights (0.8 = keep 80% of old weight)

def calculate_initial_temperature(current_sol, actions, dist_matrix, cust_addr_idx, cust_arrays, vehicles_df, num_samples=100, target_prob=0.5):
    """
    Estimates the starting temperature for Simulated Annealing by randomly 
    sampling moves and calculating the average degradation in cost.
    """
    print(f"Sampling {num_samples} moves to calculate initial temperature...")
    positive_deltas = []
    
    for _ in range(num_samples):
        # Pick a random action
        action_idx = random.randint(0, len(actions) - 1)
        d_op, r_op, _ = actions[action_idx]
        
        # Apply Destroy
        destroyed = d_op(
            current_sol,
            distance_matrix_array=dist_matrix,
            customer_addr_idx=cust_addr_idx,
            customer_arrays=cust_arrays
        )
        
        # Apply Repair
        repaired = r_op(
            destroyed,
            distance_matrix_array=dist_matrix,
            customer_addr_idx=cust_addr_idx,
            customer_arrays=cust_arrays,
            vehicles_df=vehicles_df,
            neighbor_sets=None
        )
        
        # Evaluate
        new_cost = evaluate_solution(repaired, dist_matrix, cust_addr_idx)
        delta = new_cost - current_sol._cost
        
        # We only care about worsening moves for the calculation
        if delta > 0:
            positive_deltas.append(delta)
            
    if not positive_deltas:
        # Fallback in the highly unlikely event no worse moves were generated
        print("Warning: No worse moves found during sampling. Defaulting temp to 100.0")
        return 100.0 
        
    avg_delta = sum(positive_deltas) / len(positive_deltas)
    
    # Calculate initial T using: T = -avg_delta / ln(P)
    initial_temp = -avg_delta / math.log(target_prob)
    
    return initial_temp

# ---------------------------------------------------------
#  MAIN ALNS LOOP (RRT)
# ---------------------------------------------------------

def run_alns(instance_file):

    # Load data from Solomon .txt instance
    customers_df, vehicles_df, dist_matrix, cust_addr_idx, cust_arrays = load_raw_solomon_data(instance_file)

    actions = build_actions()

    num_customers = len(customers_df['customer_id'])
    num_real_vehicles = int(vehicles_df['num_vehicles'])
    neighbor_sets = None

    # Initialize roulette wheel weights (one weight per composite action)
    action_weights = np.ones(NUM_ACTIONS)
    
    current_sol = create_initial_solution(num_customers, num_real_vehicles)
    evaluate_solution(current_sol, dist_matrix, cust_addr_idx)
    
    best_sol = current_sol.copy()
    best_sol._cost = current_sol._cost
    best_found_at = 0

    print(f"Initial Cost: {current_sol._cost:.2f}")

    # --- DYNAMIC TEMPERATURE CALCULATION ---
    # 1. Calculate the starting temperature based on average move degradation
    temperature = calculate_initial_temperature(
        current_sol, actions, dist_matrix, cust_addr_idx, cust_arrays, vehicles_df, 
        num_samples=100, target_prob=SA_TARGET_PROB
    )

    # 2. Calculate the exact cooling rate needed to reach SA_END_TEMP by MAX_ITERATIONS
    # Formula: alpha = (T_final / T_initial) ** (1 / N)
    cooling_rate = (SA_END_TEMP / temperature) ** (1 / MAX_ITERATIONS)

    print(f"Starting Main Loop (Simulated Annealing).")
    print(f"Start Temp: {temperature:.2f} | End Temp Target: {SA_END_TEMP} | Cooling Rate: {cooling_rate:.6f}")
    
    for it in range(MAX_ITERATIONS):

        # Select composite action using roulette wheel
        action_probs = action_weights / action_weights.sum()
        action_idx = np.random.choice(NUM_ACTIONS, p=action_probs)
        d_op, r_op, label = actions[action_idx]

        destroyed = d_op(
            current_sol,
            distance_matrix_array=dist_matrix,
            customer_addr_idx=cust_addr_idx,
            customer_arrays=cust_arrays
        )

        repaired = r_op(
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
            best_found_at = it
            print(f"Iter {it} [New Best]: {new_cost:.2f} (Vehicles: {sum(1 for r in best_sol.routes[:-1] if r)})")
            
        elif new_cost < current_cost:
            accepted = True
            reward = SCORE_BETTER_THAN_CURRENT
            
        else:
            # --- SIMULATED ANNEALING ACCEPTANCE CRITERIA ---
            delta_c = new_cost - current_cost
            # Prevent math overflow if temperature gets incredibly close to 0
            if temperature > 1e-6:
                probability = math.exp(-delta_c / temperature)
            else:
                probability = 0
                
            if random.random() < probability:
                accepted = True
                reward = SCORE_ACCEPTED_WORSE
            
        if accepted:
            current_sol = repaired

        # Update action weight based on performance (roulette wheel)
        action_weights[action_idx] = WEIGHT_DECAY * action_weights[action_idx] + (1 - WEIGHT_DECAY) * reward
        temperature *= cooling_rate
            
        if (it + 1) % SEGMENT_SIZE == 0:
            print(f"--- Iter {it+1} | Temp: {temperature:.2f} | Best: {best_sol._cost:.2f} | Cur: {current_sol._cost:.2f} ---")

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
    ALGORITHM = "RRT"

    # --- Add instance paths here ---
    INSTANCES = [
        os.path.join(project_root, 'Benchmark', 'data', 'homberger_600', 'C1_6_1.TXT'),
        os.path.join(project_root, 'Benchmark', 'data', 'homberger_600', 'R1_6_1.TXT'),
        os.path.join(project_root, 'Benchmark', 'data', 'homberger_600', 'RC1_6_1.TXT'),
    ]

    if not INSTANCES:
        print("No instances specified. Add paths to the INSTANCES list.")
        sys.exit(1)

    output_csv = os.path.join(project_root, 'logs', 'SA_results_600.csv')
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