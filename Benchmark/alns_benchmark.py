import pandas as pd
import numpy as np
import random
import copy
import math
from feasibility import (
    load_vrp_data,
    check_time_window_feasibility
)
from cost import calculate_route_cost

# --- Configuration ---
DUMMY_VEHICLE_NAME = 'dummy'
DUMMY_PENALTY = 10000.0
MAX_ITERATIONS = 1500
SEGMENT_SIZE = 50  # Update weights every 50 iterations

# ALNS Parameters
REACTION_FACTOR = 0.1  # (r) How quickly weights change
START_TEMPERATURE = 200.0
COOLING_RATE = 0.997

# Scoring (Sigma values)
SCORE_NEW_GLOBAL_BEST = 33
SCORE_BETTER_THAN_CURRENT = 9
SCORE_ACCEPTED_WORSE = 13
SCORE_REJECTED = 0

class Solution:
    def __init__(self, routes, vehicles):
        self.routes = [r[:] for r in routes]  # Deep copy routes
        self.vehicles = vehicles
        self._cost = None
    
    def copy(self):
        return Solution(self.routes, self.vehicles)

    def get_unassigned(self):
        # Last route is always dummy
        return self.routes[-1]

def evaluate_solution(solution, distance_matrix_array, customer_addr_idx, depot_idx=0):
    """Calculates total cost including penalties for dummy vehicle."""
    total_cost = 0.0
    
    # Real routes
    for i, route in enumerate(solution.routes[:-1]):
        if route:
            total_cost += calculate_route_cost(
                route, customer_addr_idx, distance_matrix_array, depot_idx
            )
    
    # Dummy route penalty
    unassigned_count = len(solution.routes[-1])
    total_cost += unassigned_count * DUMMY_PENALTY
    
    solution._cost = total_cost
    return total_cost

# ---------------------------------------------------------
#  HELPER FUNCTIONS
# ---------------------------------------------------------

def precompute_nearest_neighbors(distance_matrix_array, num_neighbors=20):
    """
    Returns a list of sets, where index i contains the set of nearest neighbors for customer i.
    Used for granular search (pruning).
    """
    # Argsort gives us the indices of the closest nodes
    # We skip column 0 (distance to self is 0) and take top K
    neighbor_indices = np.argsort(distance_matrix_array, axis=1)[:, 1:num_neighbors+1]
    
    # Convert to sets for O(1) lookup
    neighbors = [set(row) for row in neighbor_indices]
    return neighbors

# ---------------------------------------------------------
#  OPERATORS (Destroy & Repair)
# ---------------------------------------------------------

def random_removal(solution, num_to_remove, **kwargs):
    """Randomly removes N customers."""
    new_sol = solution.copy()
    
    candidates = []
    for r_idx, route in enumerate(new_sol.routes[:-1]):
        for c_idx, cust in enumerate(route):
            candidates.append((r_idx, cust))
            
    if not candidates:
        return new_sol

    num_to_remove = min(len(candidates), num_to_remove)
    to_remove = random.sample(candidates, num_to_remove)
    
    removed_customers = set(x[1] for x in to_remove)
    
    for i in range(len(new_sol.routes) - 1):
        new_sol.routes[i] = [c for c in new_sol.routes[i] if c not in removed_customers]
        
    new_sol.routes[-1].extend(list(removed_customers))
    return new_sol

def worst_removal(solution, num_to_remove, distance_matrix_array, customer_addr_idx, depot_idx=0, **kwargs):
    """Removes customers that contribute the most to the distance cost."""
    new_sol = solution.copy()
    savings_list = [] # (cost_diff, route_idx, customer_val)

    for r_idx, route in enumerate(new_sol.routes[:-1]):
        if len(route) == 0: continue
        
        current_cost = calculate_route_cost(route, customer_addr_idx, distance_matrix_array, depot_idx)
        
        for i, cust in enumerate(route):
            temp_route = route[:i] + route[i+1:]
            new_cost = calculate_route_cost(temp_route, customer_addr_idx, distance_matrix_array, depot_idx)
            diff = current_cost - new_cost
            savings_list.append((diff, r_idx, cust))
    
    savings_list.sort(key=lambda x: x[0], reverse=True)
    
    targets = set()
    count = 0
    for _, _, cust in savings_list:
        targets.add(cust)
        count += 1
        if count >= num_to_remove:
            break
            
    for i in range(len(new_sol.routes) - 1):
        new_sol.routes[i] = [c for c in new_sol.routes[i] if c not in targets]
        
    new_sol.routes[-1].extend(list(targets))
    return new_sol

def greedy_insertion(solution, distance_matrix_array, customer_addr_idx, customer_arrays, vehicles_df, neighbor_sets, depot_idx=0, **kwargs):
    """
    Optimized Greedy Insertion:
    1. Uses O(1) Delta Cost calculation.
    2. Uses 'Lazy' feasibility checks.
    3. Uses Neighbor Pruning (Granular Search).
    """
    new_sol = solution.copy()
    unassigned = list(new_sol.routes[-1])
    new_sol.routes[-1] = [] # Clear dummy
    
    random.shuffle(unassigned)
    
    for cust in unassigned:
        best_cost_increase = float('inf')
        best_pos = None # (route_idx, index)
        
        cust_addr = customer_addr_idx[cust-1]
        cust_demand = customer_arrays['demand'][cust-1]
        
        # Optimization: Only look for insertions after these nodes (plus Depot)
        allowed_prev_nodes = neighbor_sets[cust_addr]
        
        for r_idx in range(len(new_sol.routes) - 1):
            route = new_sol.routes[r_idx]
            vehicle_name = new_sol.vehicles[r_idx]
            capacity = vehicles_df.loc[vehicle_name, 'capacity']
            
            # 1. Fast Capacity Check
            current_route_demand = sum(customer_arrays['demand'][c-1] for c in route)
            if current_route_demand + cust_demand > capacity:
                continue
                
            # Map route to matrix indices [Depot, c1, c2..., Depot]
            route_addrs = [depot_idx] + [customer_addr_idx[c-1] for c in route] + [depot_idx]
            
            for i in range(len(route) + 1):
                prev_node = route_addrs[i]
                
                # --- PRUNING ---
                # Skip if previous node is not the depot AND not a neighbor
                if prev_node != depot_idx and prev_node not in allowed_prev_nodes:
                    continue
                # ---------------

                next_node = route_addrs[i+1]
                
                # DELTA COST (O(1))
                added = distance_matrix_array[prev_node, cust_addr] + \
                        distance_matrix_array[cust_addr, next_node]
                removed = distance_matrix_array[prev_node, next_node]
                
                marginal_cost = added - removed
                
                # LAZY CHECK: Only check complex time windows if this is the best move so far
                if marginal_cost < best_cost_increase:
                    candidate_route = route[:i] + [cust] + route[i:]
                    
                    if check_time_window_feasibility(candidate_route, distance_matrix_array, customer_addr_idx, customer_arrays, depot_idx):
                        best_cost_increase = marginal_cost
                        best_pos = (r_idx, i)
        
        if best_pos:
            r_idx, idx = best_pos
            new_sol.routes[r_idx].insert(idx, cust)
        else:
            new_sol.routes[-1].append(cust)
            
    return new_sol

def regret_insertion(solution, distance_matrix_array, customer_addr_idx, customer_arrays, vehicles_df, neighbor_sets, depot_idx=0, **kwargs):
    """
    Optimized 2-Regret Insertion with Delta Cost, Lazy Checks, and Neighbor Pruning.
    """
    new_sol = solution.copy()
    unassigned = list(new_sol.routes[-1])
    new_sol.routes[-1] = []
    
    # Cache capacities
    capacities = [vehicles_df.loc[v, 'capacity'] for v in new_sol.vehicles[:-1]]
    
    while unassigned:
        best_regret_score = -1
        best_customer = None
        best_insertion_info = None
        
        candidates_found = False
        
        # Optimization: Shuffle unassigned to avoid bias in ties
        random.shuffle(unassigned)
        
        for cust in unassigned:
            cust_addr = customer_addr_idx[cust-1]
            cust_demand = customer_arrays['demand'][cust-1]
            allowed_prev_nodes = neighbor_sets[cust_addr]
            
            feasible_insertions = [] # (marginal_cost, route_idx, index)
            
            for r_idx in range(len(new_sol.routes) - 1):
                route = new_sol.routes[r_idx]
                
                # Fast Capacity Check
                current_demand = sum(customer_arrays['demand'][c-1] for c in route)
                if current_demand + cust_demand > capacities[r_idx]:
                    continue
                
                route_addrs = [depot_idx] + [customer_addr_idx[c-1] for c in route] + [depot_idx]
                
                for i in range(len(route) + 1):
                    prev_node = route_addrs[i]
                    
                    # --- PRUNING ---
                    if prev_node != depot_idx and prev_node not in allowed_prev_nodes:
                        continue
                    # ---------------
                    
                    next_node = route_addrs[i+1]
                    
                    # Delta Cost
                    added = distance_matrix_array[prev_node, cust_addr] + \
                            distance_matrix_array[cust_addr, next_node]
                    removed = distance_matrix_array[prev_node, next_node]
                    marginal_cost = added - removed
                    
                    feasible_insertions.append((marginal_cost, r_idx, i))

            # Sort by cost (Cheapest first)
            feasible_insertions.sort(key=lambda x: x[0])
            
            # Lazy Feasibility Check (Check only top 2 valid)
            valid_insertions = []
            for cost, r_idx, i in feasible_insertions:
                if len(valid_insertions) >= 2: 
                    break
                
                route = new_sol.routes[r_idx]
                candidate_route = route[:i] + [cust] + route[i:]
                
                if check_time_window_feasibility(candidate_route, distance_matrix_array, customer_addr_idx, customer_arrays, depot_idx):
                    valid_insertions.append((cost, r_idx, i))
            
            if not valid_insertions:
                continue
                
            candidates_found = True
            
            # Calculate Regret
            if len(valid_insertions) == 1:
                regret = float('inf') 
                chosen_insertion = valid_insertions[0]
            else:
                best = valid_insertions[0]
                second = valid_insertions[1]
                regret = second[0] - best[0]
                chosen_insertion = best
            
            if regret > best_regret_score:
                best_regret_score = regret
                best_customer = cust
                best_insertion_info = (chosen_insertion[1], chosen_insertion[2])
        
        if not candidates_found:
            new_sol.routes[-1].extend(unassigned)
            break
            
        r_idx, idx = best_insertion_info
        new_sol.routes[r_idx].insert(idx, best_customer)
        unassigned.remove(best_customer)
        
    return new_sol

# ---------------------------------------------------------
#  ADAPTIVE ENGINE
# ---------------------------------------------------------

def select_operator(weights):
    """Roulette wheel selection."""
    total = sum(weights)
    if total == 0:
        return random.choice(range(len(weights)))
    probs = [w/total for w in weights]
    return np.random.choice(len(weights), p=probs)

def update_weights(weights, scores, attempts):
    """Updates weights based on performance."""
    for i in range(len(weights)):
        if attempts[i] > 0:
            avg_score = scores[i] / attempts[i]
            weights[i] = (1 - REACTION_FACTOR) * weights[i] + REACTION_FACTOR * avg_score

def create_initial_solution(num_customers, num_real_vehicles):
    """Basic solution: All to dummy."""
    routes = [[] for _ in range(num_real_vehicles)]
    routes.append(list(range(1, num_customers + 1)))
    vehicles = ['Standard'] * num_real_vehicles + [DUMMY_VEHICLE_NAME]
    return Solution(routes, vehicles)

# ---------------------------------------------------------
#  MAIN ALNS LOOP
# ---------------------------------------------------------

def run_alns():
    customers_df, vehicles_df, _, dist_matrix, cust_addr_idx, cust_arrays = load_vrp_data()
    
    num_customers = len(customers_df)
    num_real_vehicles = int(vehicles_df.loc['Standard', 'num_vehicles'])
    
    # Precompute Neighbors (Granular Search)
    neighbor_sets = precompute_nearest_neighbors(dist_matrix, num_neighbors=25)
    
    # 1. Operators & Weights
    destroy_ops = [random_removal, worst_removal]
    repair_ops = [greedy_insertion, regret_insertion]
    
    d_weights = [1.0] * len(destroy_ops)
    r_weights = [1.0] * len(repair_ops)
    
    d_scores = [0.0] * len(destroy_ops)
    r_scores = [0.0] * len(repair_ops)
    
    d_attempts = [0] * len(destroy_ops)
    r_attempts = [0] * len(repair_ops)
    
    # 2. Initialization
    current_sol = create_initial_solution(num_customers, num_real_vehicles)
    evaluate_solution(current_sol, dist_matrix, cust_addr_idx)
    
    best_sol = current_sol.copy()
    best_sol._cost = current_sol._cost
    
    curr_temp = START_TEMPERATURE
    
    print(f"Initial Cost: {current_sol._cost:.2f}")
    
    # 3. Main Loop
    for it in range(MAX_ITERATIONS):
        # Select Operators
        d_idx = select_operator(d_weights)
        r_idx = select_operator(r_weights)
        
        # Apply Destroy
        n_remove = random.randint(int(num_customers * 0.1), int(num_customers * 0.4))
        
        destroyed_sol = destroy_ops[d_idx](
            current_sol, 
            n_remove, 
            distance_matrix_array=dist_matrix, 
            customer_addr_idx=cust_addr_idx
        )
        
        # Apply Repair (Pass neighbor_sets)
        repaired_sol = repair_ops[r_idx](
            destroyed_sol, 
            distance_matrix_array=dist_matrix, 
            customer_addr_idx=cust_addr_idx,
            customer_arrays=cust_arrays,
            vehicles_df=vehicles_df,
            neighbor_sets=neighbor_sets
        )
        
        # Evaluate
        new_cost = evaluate_solution(repaired_sol, dist_matrix, cust_addr_idx)
        current_cost = current_sol._cost
        best_cost = best_sol._cost
        
        # Acceptance (Simulated Annealing)
        accepted = False
        score = SCORE_REJECTED
        
        delta = new_cost - current_cost
        
        if delta < 0:
            accepted = True
            if new_cost < best_cost:
                score = SCORE_NEW_GLOBAL_BEST
                best_sol = repaired_sol.copy()
                best_sol._cost = new_cost
                print(f"Iter {it}: New Best {new_cost:.2f} (Unassigned: {len(best_sol.routes[-1])})")
            else:
                score = SCORE_BETTER_THAN_CURRENT
        else:
            prob = math.exp(-delta / curr_temp)
            if random.random() < prob:
                accepted = True
                score = SCORE_ACCEPTED_WORSE
                
        if accepted:
            current_sol = repaired_sol
            
        # Update Scores
        d_scores[d_idx] += score
        r_scores[r_idx] += score
        d_attempts[d_idx] += 1
        r_attempts[r_idx] += 1
        
        # Segment Update
        if (it + 1) % SEGMENT_SIZE == 0:
            update_weights(d_weights, d_scores, d_attempts)
            update_weights(r_weights, r_scores, r_attempts)
            
            d_scores = [0.0] * len(destroy_ops)
            r_scores = [0.0] * len(repair_ops)
            d_attempts = [0] * len(destroy_ops)
            r_attempts = [0] * len(repair_ops)
            
            print(f"--- Segment {it // SEGMENT_SIZE} Complete ---")
            print(f"Temp: {curr_temp:.2f}, Best: {best_sol._cost:.2f}")

        # Cool Down
        curr_temp *= COOLING_RATE

    # Final Output
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Best Cost: {best_sol._cost:.2f}")
    unassigned = len(best_sol.routes[-1])
    print(f"Unassigned Customers: {unassigned}")
    vehicles_used = sum(1 for route in best_sol.routes[:-1] if route)
    print(f"Vehicles Used: {vehicles_used}")
    
    print("\nRoutes:")
    for i, route in enumerate(best_sol.routes[:-1]):
        if route:
            c = calculate_route_cost(route, cust_addr_idx, dist_matrix, 0)
            print(f"Vehicle {i+1}: {route} (Cost: {c:.2f})")

if __name__ == "__main__":
    run_alns()