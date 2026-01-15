import pandas as pd
import numpy as np
import random
import copy
import math
from feasibility import (
    load_vrp_data,
    check_capacity_feasibility,
    check_time_window_feasibility
)
from cost import calculate_route_cost

# --- Configuration ---
DUMMY_VEHICLE_NAME = 'dummy'
DUMMY_PENALTY = 10000.0
MAX_ITERATIONS = 1500
SEGMENT_SIZE = 50  # Only used for logging now, not for weight updates

# ALNS Parameters
START_TEMPERATURE = 200.0
COOLING_RATE = 0.997

# Scoring (Rewards for RL)
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

# ---------------------------------------------------------
#  RL / Q-LEARNING ENGINE
# ---------------------------------------------------------

class QLearningAgent:
    def __init__(self, num_destroy, num_repair, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha      # Learning Rate
        self.gamma = gamma      # Discount Factor
        self.epsilon = epsilon  # Exploration Rate
        
        # Q-Tables: [State][Operator_Index]
        # States: 0=Early Search, 1=Mid Search, 2=Late Search
        self.q_destroy = np.zeros((3, num_destroy))
        self.q_repair = np.zeros((3, num_repair))
        
    def get_state(self, current_iter, max_iter):
        """Map search progress to a discrete state."""
        progress = current_iter / max_iter
        if progress < 0.33: return 0  # Early
        if progress < 0.66: return 1  # Mid
        return 2                      # Late

    def select_action(self, state, q_table):
        """Epsilon-Greedy Selection strategy."""
        if random.random() < self.epsilon:
            return random.randint(0, len(q_table[state]) - 1) # Explore
        else:
            # Add small noise to break ties randomly
            values = q_table[state]
            random_noise = np.random.random(values.shape) * 1e-5
            return np.argmax(values + random_noise) # Exploit

    def update(self, state, action, reward, next_state, q_table):
        """Bellman Equation Update."""
        best_next = np.max(q_table[next_state])
        current_q = q_table[state][action]
        q_table[state][action] = current_q + \
                                 self.alpha * (reward + self.gamma * best_next - current_q)

# ---------------------------------------------------------
#  HELPER FUNCTIONS
# ---------------------------------------------------------

def evaluate_solution(solution, distance_matrix_array, customer_addr_idx, depot_idx=0):
    """Calculates total cost including penalties for dummy vehicle."""
    total_cost = 0.0
    
    # Real routes
    for i, route in enumerate(solution.routes[:-1]):
        if route:
            # Using imported function from cost.py
            total_cost += calculate_route_cost(
                route, customer_addr_idx, distance_matrix_array, depot_idx
            )
    
    # Dummy route penalty
    unassigned_count = len(solution.routes[-1])
    total_cost += unassigned_count * DUMMY_PENALTY
    
    solution._cost = total_cost
    return total_cost

def precompute_nearest_neighbors(distance_matrix_array, num_neighbors=20):
    """
    Returns a list of sets, where index i contains the set of nearest neighbors for customer i.
    """
    # Argsort gives indices of closest nodes. Skip col 0 (self) and take top K.
    neighbor_indices = np.argsort(distance_matrix_array, axis=1)[:, 1:num_neighbors+1]
    neighbors = [set(row) for row in neighbor_indices]
    return neighbors

# ---------------------------------------------------------
#  LOCAL SEARCH (New!)
# ---------------------------------------------------------

def two_opt_local_search(solution, distance_matrix_array, customer_addr_idx, customer_arrays, depot_idx=0):
    """
    Applies 2-opt swaps within each route to untangle crossings.
    Checks time window feasibility before accepting swaps.
    """
    improved = False
    
    for r_idx, route in enumerate(solution.routes[:-1]):
        if len(route) < 3: continue
        
        # Route path including depot for calculation: [0, c1, c2, ..., 0]
        # But we only modify the 'route' list (indices of customers)
        
        best_route_cost = calculate_route_cost(route, customer_addr_idx, distance_matrix_array, depot_idx)
        route_improved = True
        
        while route_improved:
            route_improved = False
            # Iterate over all possible segments to reverse
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    if j - i == 1: continue # Skip adjacent pairs (no change)
                    
                    # Create candidate by reversing segment i:j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    
                    # 1. Fast Cost Check
                    new_cost = calculate_route_cost(new_route, customer_addr_idx, distance_matrix_array, depot_idx)
                    
                    if new_cost < best_route_cost - 1e-3: # Tolerance
                        # 2. Feasibility Check (Capacity is unchanged by swap, only check Time)
                        if check_time_window_feasibility(new_route, distance_matrix_array, customer_addr_idx, customer_arrays, depot_idx):
                            route[:] = new_route # Update in place
                            best_route_cost = new_cost
                            route_improved = True
                            improved = True
                            break # Restart scan for this route
                if route_improved: break
                
    return improved

# ---------------------------------------------------------
#  OPERATORS (Destroy)
# ---------------------------------------------------------

def random_removal(solution, num_to_remove, **kwargs):
    """Randomly removes N customers."""
    new_sol = solution.copy()
    candidates = []
    for r_idx, route in enumerate(new_sol.routes[:-1]):
        for cust in route:
            candidates.append((r_idx, cust))
            
    if not candidates: return new_sol

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
        if not route: continue
        current_cost = calculate_route_cost(route, customer_addr_idx, distance_matrix_array, depot_idx)
        
        for i, cust in enumerate(route):
            temp_route = route[:i] + route[i+1:]
            new_cost = calculate_route_cost(temp_route, customer_addr_idx, distance_matrix_array, depot_idx)
            diff = current_cost - new_cost
            savings_list.append((diff, r_idx, cust))
    
    savings_list.sort(key=lambda x: x[0], reverse=True)
    
    targets = set()
    for _, _, cust in savings_list[:num_to_remove]:
        targets.add(cust)
            
    for i in range(len(new_sol.routes) - 1):
        new_sol.routes[i] = [c for c in new_sol.routes[i] if c not in targets]
        
    new_sol.routes[-1].extend(list(targets))
    return new_sol

def cluster_removal(solution, num_to_remove, distance_matrix_array, customer_addr_idx, **kwargs):
    """
    Spatial Removal: Picks a random center customer and removes its N nearest neighbors.
    Great for restructuring specific geographic regions.
    """
    new_sol = solution.copy()
    
    # 1. Pick random center
    candidates = []
    for r in new_sol.routes[:-1]:
        candidates.extend(r)
    
    if not candidates: return new_sol
    
    center_cust = random.choice(candidates)
    center_idx = customer_addr_idx[center_cust-1]
    
    # 2. Calculate distances from center to all other present customers
    distances = []
    for cust in candidates:
        if cust == center_cust: continue
        idx = customer_addr_idx[cust-1]
        dist = distance_matrix_array[center_idx, idx]
        distances.append((dist, cust))
    
    # 3. Sort by distance and pick top N
    distances.sort(key=lambda x: x[0])
    
    targets = {center_cust}
    for _, cust in distances[:num_to_remove-1]:
        targets.add(cust)
    
    # 4. Remove
    for i in range(len(new_sol.routes) - 1):
        new_sol.routes[i] = [c for c in new_sol.routes[i] if c not in targets]
        
    new_sol.routes[-1].extend(list(targets))
    return new_sol

# ---------------------------------------------------------
#  OPERATORS (Repair)
# ---------------------------------------------------------

def greedy_insertion(solution, distance_matrix_array, customer_addr_idx, customer_arrays, vehicles_df, neighbor_sets, depot_idx=0, **kwargs):
    """Optimized Greedy Insertion with Pruning and Lazy Checks."""
    new_sol = solution.copy()
    unassigned = list(new_sol.routes[-1])
    new_sol.routes[-1] = []
    random.shuffle(unassigned)
    
    for cust in unassigned:
        best_cost_increase = float('inf')
        best_pos = None # (route_idx, index)
        
        cust_addr = customer_addr_idx[cust-1]
        cust_demand = customer_arrays['demand'][cust-1]
        allowed_prev_nodes = neighbor_sets[cust_addr]
        
        for r_idx in range(len(new_sol.routes) - 1):
            route = new_sol.routes[r_idx]
            vehicle_name = new_sol.vehicles[r_idx]
            
            # Fast Capacity Check
            # Note: We reconstruct route indices for check_capacity_feasibility if needed, 
            # but simple sum check is faster here for single insertion
            if not check_capacity_feasibility(route + [cust], vehicle_name, vehicles_df, customer_arrays):
                continue
                
            route_addrs = [depot_idx] + [customer_addr_idx[c-1] for c in route] + [depot_idx]
            
            for i in range(len(route) + 1):
                prev_node = route_addrs[i]
                if prev_node != depot_idx and prev_node not in allowed_prev_nodes:
                    continue # Pruning

                next_node = route_addrs[i+1]
                
                # Delta Cost O(1)
                marginal_cost = (distance_matrix_array[prev_node, cust_addr] + 
                                 distance_matrix_array[cust_addr, next_node] - 
                                 distance_matrix_array[prev_node, next_node])
                
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
    """2-Regret Insertion with Pruning."""
    new_sol = solution.copy()
    unassigned = list(new_sol.routes[-1])
    new_sol.routes[-1] = []
    
    while unassigned:
        best_regret_score = -1
        best_customer = None
        best_insertion_info = None
        candidates_found = False
        
        random.shuffle(unassigned)
        
        for cust in unassigned:
            cust_addr = customer_addr_idx[cust-1]
            allowed_prev_nodes = neighbor_sets[cust_addr]
            feasible_insertions = [] 
            
            for r_idx in range(len(new_sol.routes) - 1):
                route = new_sol.routes[r_idx]
                vehicle_name = new_sol.vehicles[r_idx]

                if not check_capacity_feasibility(route + [cust], vehicle_name, vehicles_df, customer_arrays):
                    continue
                
                route_addrs = [depot_idx] + [customer_addr_idx[c-1] for c in route] + [depot_idx]
                for i in range(len(route) + 1):
                    prev_node = route_addrs[i]
                    if prev_node != depot_idx and prev_node not in allowed_prev_nodes: continue

                    next_node = route_addrs[i+1]
                    marginal_cost = (distance_matrix_array[prev_node, cust_addr] + 
                                     distance_matrix_array[cust_addr, next_node] - 
                                     distance_matrix_array[prev_node, next_node])
                    feasible_insertions.append((marginal_cost, r_idx, i))

            feasible_insertions.sort(key=lambda x: x[0])
            
            # Lazy check top 2
            valid_insertions = []
            for cost, r_idx, i in feasible_insertions:
                if len(valid_insertions) >= 2: break
                candidate_route = new_sol.routes[r_idx][:i] + [cust] + new_sol.routes[r_idx][i:]
                if check_time_window_feasibility(candidate_route, distance_matrix_array, customer_addr_idx, customer_arrays, depot_idx):
                    valid_insertions.append((cost, r_idx, i))
            
            if not valid_insertions: continue
            candidates_found = True
            
            # Regret calculation
            if len(valid_insertions) == 1:
                regret = float('inf') 
                chosen = valid_insertions[0]
            else:
                regret = valid_insertions[1][0] - valid_insertions[0][0]
                chosen = valid_insertions[0]
            
            if regret > best_regret_score:
                best_regret_score = regret
                best_customer = cust
                best_insertion_info = (chosen[1], chosen[2])
        
        if not candidates_found:
            new_sol.routes[-1].extend(unassigned)
            break
            
        r_idx, idx = best_insertion_info
        new_sol.routes[r_idx].insert(idx, best_customer)
        unassigned.remove(best_customer)
        
    return new_sol

# ---------------------------------------------------------
#  MAIN ALNS LOOP
# ---------------------------------------------------------

def create_initial_solution(num_customers, num_real_vehicles):
    """Basic solution: All to dummy."""
    routes = [[] for _ in range(num_real_vehicles)]
    routes.append(list(range(1, num_customers + 1)))
    vehicles = ['Standard'] * num_real_vehicles + [DUMMY_VEHICLE_NAME]
    return Solution(routes, vehicles)

def run_alns():
    customers_df, vehicles_df, _, dist_matrix, cust_addr_idx, cust_arrays = load_vrp_data()
    
    num_customers = len(customers_df)
    # Assuming 'Standard' vehicle type exists, defaulting to 10 if not specified
    try:
        num_real_vehicles = int(vehicles_df.loc['Standard', 'num_vehicles'])
    except KeyError:
        print("Warning: 'Standard' vehicle not found, defaulting to 10 vehicles.")
        num_real_vehicles = 10
    
    print("Precomputing nearest neighbors for granular search...")
    neighbor_sets = precompute_nearest_neighbors(dist_matrix, num_neighbors=25)
    
    # 1. Operators definition
    destroy_ops = [random_removal, worst_removal, cluster_removal]
    repair_ops = [greedy_insertion, regret_insertion]
    
    # 2. Initialize Q-Learning Agent
    agent = QLearningAgent(len(destroy_ops), len(repair_ops))
    
    # 3. Initialization
    current_sol = create_initial_solution(num_customers, num_real_vehicles)
    evaluate_solution(current_sol, dist_matrix, cust_addr_idx)
    
    best_sol = current_sol.copy()
    best_sol._cost = current_sol._cost
    curr_temp = START_TEMPERATURE
    
    print(f"Initial Cost: {current_sol._cost:.2f}")
    
    # 4. Main Loop
    for it in range(MAX_ITERATIONS):
        # Determine State
        state = agent.get_state(it, MAX_ITERATIONS)
        
        # Select Actions (Operators)
        d_idx = agent.select_action(state, agent.q_destroy)
        r_idx = agent.select_action(state, agent.q_repair)
        
        # Apply Destroy
        n_remove = random.randint(int(num_customers * 0.1), int(num_customers * 0.4))
        destroyed_sol = destroy_ops[d_idx](
            current_sol, 
            n_remove, 
            distance_matrix_array=dist_matrix, 
            customer_addr_idx=cust_addr_idx
        )
        
        # Apply Repair
        repaired_sol = repair_ops[r_idx](
            destroyed_sol, 
            distance_matrix_array=dist_matrix, 
            customer_addr_idx=cust_addr_idx,
            customer_arrays=cust_arrays,
            vehicles_df=vehicles_df,
            neighbor_sets=neighbor_sets
        )
        
        # Apply Local Search (Improvement Step)
        two_opt_local_search(
            repaired_sol, 
            dist_matrix, 
            cust_addr_idx, 
            cust_arrays
        )
        
        # Evaluate
        new_cost = evaluate_solution(repaired_sol, dist_matrix, cust_addr_idx)
        current_cost = current_sol._cost
        best_cost = best_sol._cost
        
        # Acceptance & Reward Calculation
        accepted = False
        reward = SCORE_REJECTED
        delta = new_cost - current_cost
        
        if delta < 0:
            accepted = True
            if new_cost < best_cost:
                reward = SCORE_NEW_GLOBAL_BEST
                best_sol = repaired_sol.copy()
                best_sol._cost = new_cost
                print(f"Iter {it}: New Best {new_cost:.2f} (Unassigned: {len(best_sol.routes[-1])})")
            else:
                reward = SCORE_BETTER_THAN_CURRENT
        else:
            prob = math.exp(-delta / curr_temp)
            if random.random() < prob:
                accepted = True
                reward = SCORE_ACCEPTED_WORSE
        
        if accepted:
            current_sol = repaired_sol
            
        # RL Update (Learn from the result)
        next_state = agent.get_state(it + 1, MAX_ITERATIONS)
        agent.update(state, d_idx, reward, next_state, agent.q_destroy)
        agent.update(state, r_idx, reward, next_state, agent.q_repair)

        # Logging
        if (it + 1) % SEGMENT_SIZE == 0:
            print(f"--- Segment {it // SEGMENT_SIZE} | Temp: {curr_temp:.2f} | Best: {best_sol._cost:.2f} ---")

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