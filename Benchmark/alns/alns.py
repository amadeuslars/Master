import random
from utils.utils import evaluate_solution, create_initial_solution, precompute_nearest_neighbors, load_vrp_data
from utils.operators import random_removal, greedy_insertion
from utils.feasibility import load_vrp_data


# --- Configuration ---
DUMMY_VEHICLE_NAME = 'dummy'
DUMMY_PENALTY = 10000.0
MAX_ITERATIONS = 10000
SEGMENT_SIZE = 50 

def run_alns():
    customers_df, vehicles_df, _, dist_matrix, cust_addr_idx, cust_arrays = load_vrp_data()
    
    destroy_ops = [random_removal]
    repair_ops = [greedy_insertion]

    num_customers = len(customers_df)
    try:
        num_real_vehicles = int(vehicles_df.loc['Standard', 'num_vehicles'])
    except KeyError:
        num_real_vehicles = 25 

    neighbor_sets = precompute_nearest_neighbors(dist_matrix, num_neighbors=10)
    
    current_sol = create_initial_solution(num_customers, num_real_vehicles)
    evaluate_solution(current_sol, dist_matrix, cust_addr_idx)

    best_sol = current_sol.copy()
    best_sol._cost = current_sol._cost

    print(f"Initial Cost: {current_sol._cost:.2f}")

    for it in range(MAX_ITERATIONS):
        # Simple destroy/repair
        n_remove = random.randint(int(num_customers * 0.1), int(num_customers * 0.2))
        destroyed = destroy_ops[0](current_sol, n_remove, distance_matrix_array=dist_matrix, customer_addr_idx=cust_addr_idx, customer_arrays=cust_arrays)
        repaired = repair_ops[0](destroyed, distance_matrix_array=dist_matrix, customer_addr_idx=cust_addr_idx, customer_arrays=cust_arrays, vehicles_df=vehicles_df, neighbor_sets=neighbor_sets)
        evaluate_solution(repaired, dist_matrix, cust_addr_idx)

        # Accept only if better
        if repaired._cost < current_sol._cost:
            current_sol = repaired
            if repaired._cost < best_sol._cost:
                best_sol = repaired.copy()
                best_sol._cost = repaired._cost
                print(f"Iter {it} [New Best]: {repaired._cost:.2f}")
        
        if (it + 1) % SEGMENT_SIZE == 0:
            print(f"--- Iter {it+1} | Best: {best_sol._cost:.2f} | Cur: {current_sol._cost:.2f} ---")

    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(f"Best Cost: {best_sol._cost:.2f}")
    
    print("Routes:")
    for i, r in enumerate(best_sol.routes[:-1]):
        if r:
            load = sum(cust_arrays['demand'][c-1] for c in r)
            print(f"V{i+1}: {r} | Load: {load}")


if __name__ == "__main__":
    run_alns()