import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import load_vrp_data, generate_initial_solution, evaluate_solution
from utils.operators import random_removal, greedy_insertion

# --- Configuration ---

MAX_ITERATIONS = 10000       


def run_alns():
    
    customers_dict, vehicles_dict, vehicle_names, time_matrix_array, depot_idx, addr_idx, customer_arrays = load_vrp_data()

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

    # Generate initial dummy solution with correct vehicle names
    current_sol = generate_initial_solution(customers_dict, vehicle_names=sorted_vehicle_names)
    evaluate_solution(current_sol, addr_idx, time_matrix_array, depot_idx)
    best_sol = current_sol.copy()

    print(f"Initial Cost: {current_sol.cost:.2f}")

    print("\n--- Solution object passed to operators ---")
    print(f"  type: {type(current_sol)}")
    print(f"  .routes: {current_sol.routes}")
    print(f"  .vehicles: {current_sol.vehicles}")
    print(f"  .cost: {current_sol.cost}")
    print(f"  .unassigned: {current_sol.unassigned}")
    print()

    start_time = time.time()

    for i in range(MAX_ITERATIONS):
        n_remove = max(0.02* int(sum(len(r) for r in current_sol.routes)), int(sum(len(r) for r in current_sol.routes)*0.4))
        destroyed_sol = random_removal(current_sol, n_remove)

        repaired_sol = greedy_insertion(
            destroyed_sol,
            time_matrix_array,
            addr_idx,
            customer_arrays,
            vehicles_dict,
            None,  # neighbor_sets placeholder
            depot_idx=depot_idx,
            temperature=1.0,
            compatible_ppls_set=compatible_ppls_set
        )

        evaluate_solution(repaired_sol, addr_idx, time_matrix_array, depot_idx)

        if repaired_sol.cost < current_sol.cost:
            current_sol = repaired_sol
            if current_sol.cost < best_sol.cost:
                best_sol = current_sol.copy()
                print(f"Iter {i}: New Best Cost={best_sol.cost:.2f}")
    
    print(f"Total Runtime: {time.time() - start_time:.2f}s")
    print("\n--- Final Results ---")
    print(f"Best Cost: {best_sol.cost:.2f} hours")
    print(f"Vehicles used: {sum(1 for i, r in enumerate(best_sol.routes) if r and best_sol.vehicles[i] != 'dummy')}")
    print()
    cids = customers_dict['customer_id']
    for i, (route, veh) in enumerate(zip(best_sol.routes, best_sol.vehicles)):
        if route:
            kundenr = [int(cids[c - 1]) for c in route]
            print(f"  {veh}: {kundenr}")
    if best_sol.routes[-1]:
        unassigned_kundenr = [int(cids[c - 1]) for c in best_sol.routes[-1]]
        print(f"\n  Unassigned ({len(best_sol.routes[-1])}): {unassigned_kundenr}")

    return best_sol

if __name__ == "__main__":
    sol = run_alns()
    