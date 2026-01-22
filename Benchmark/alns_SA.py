import numpy as np
import random
import math
from feasibility import (
    load_vrp_data,
    check_capacity_feasibility,
    check_time_window_feasibility
)
from cost import calculate_route_cost
from visualization import ALNSTracker

# --- Configuration ---
DUMMY_VEHICLE_NAME = 'dummy'
DUMMY_PENALTY = 10000.0
MAX_ITERATIONS = 10000
SEGMENT_SIZE = 50 

# ALNS Parameters
START_TEMPERATURE = 200.0
WARMUP_ITERATIONS = 100   
ESCAPE_THRESHOLD = 1000

# Scoring (Rewards for RL)
SCORE_NEW_GLOBAL_BEST = 35
SCORE_BETTER_THAN_CURRENT = 10
SCORE_ACCEPTED_WORSE = 5
SCORE_REJECTED = 0

class Solution:
    def __init__(self, routes, vehicles):
        self.routes = [r[:] for r in routes] 
        self.vehicles = vehicles
        self._cost = None
    
    def copy(self):
        return Solution(self.routes, self.vehicles)

# ---------------------------------------------------------
#  RL / Q-LEARNING ENGINE
# ---------------------------------------------------------

class QLearningAgent:
    def __init__(self, num_destroy, num_repair, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha      
        self.gamma = gamma      
        self.epsilon = epsilon  
        
        # Q-Tables: [State][Operator_Index]
        # States: 0=Early, 1=Mid, 2=Late
        self.q_destroy = np.zeros((3, num_destroy))
        self.q_repair = np.zeros((3, num_repair))
        
        # Initialize with optimistic values to encourage exploration
        self.q_destroy.fill(5.0)
        self.q_repair.fill(5.0)
        
    def get_state(self, current_iter, max_iter):
        progress = current_iter / max_iter
        if progress < 0.33: return 0  
        if progress < 0.66: return 1  
        return 2                      

    def select_action(self, state, q_table):
        if random.random() < self.epsilon:
            return random.randint(0, len(q_table[state]) - 1) 
        else:
            values = q_table[state]
            random_noise = np.random.random(values.shape) * 1e-4 # Tiny noise for tie-breaking
            return np.argmax(values + random_noise)

    def update(self, state, action, reward, next_state, q_table):
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
            total_cost += calculate_route_cost(
                route, customer_addr_idx, distance_matrix_array, depot_idx
            )
    
    # Dummy route penalty
    unassigned_count = len(solution.routes[-1])
    total_cost += unassigned_count * DUMMY_PENALTY
    
    solution._cost = total_cost
    return total_cost

def precompute_nearest_neighbors(distance_matrix_array, num_neighbors=20):
    n = distance_matrix_array.shape[0]
    k = min(num_neighbors, n - 1)
    neighbors = []
    for i in range(n):
        row = distance_matrix_array[i]
        idxs = np.argpartition(row, k)[:k+1]
        idxs = idxs[idxs != i]
        if idxs.shape[0] > k:
            order = np.argsort(row[idxs])[:k]
            idxs = idxs[order]
        else:
            order = np.argsort(row[idxs])
            idxs = idxs[order]
        neighbors.append(set(map(int, idxs.tolist())))
    return neighbors

# ---------------------------------------------------------
#  LOCAL SEARCH (Tiered Strategy)
# ---------------------------------------------------------

def two_opt_local_search(solution, distance_matrix_array, customer_addr_idx, customer_arrays, depot_idx=0):
    """INTRA-ROUTE: Untangles crossings within single routes."""
    improved = False
    for r_idx, route in enumerate(solution.routes[:-1]):
        if len(route) < 3: continue
        
        best_route_cost = calculate_route_cost(route, customer_addr_idx, distance_matrix_array, depot_idx)
        route_improved = True
        
        while route_improved:
            route_improved = False
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    if j - i == 1: continue 
                    
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_cost = calculate_route_cost(new_route, customer_addr_idx, distance_matrix_array, depot_idx)
                    
                    if new_cost < best_route_cost - 1e-4:
                        if check_time_window_feasibility(new_route, distance_matrix_array, customer_addr_idx, customer_arrays, depot_idx):
                            route[:] = new_route 
                            best_route_cost = new_cost
                            route_improved = True
                            improved = True
                            break 
                if route_improved: break
    return improved

def simple_relocate(solution, distance_matrix_array, customer_addr_idx, customer_arrays, vehicles_df, depot_idx=0):
    """
    INTER-ROUTE: Moves single customers between routes.
    Runs faster than segment relocation, good for continuous balancing.
    """
    improved = False
    capacities = [vehicles_df.loc[v, 'capacity'] for v in solution.vehicles[:-1]]
    
    for r_src_idx, src_route in enumerate(solution.routes[:-1]):
        if not src_route: continue
        
        for i, cust in enumerate(src_route):
            cust_demand = customer_arrays['demand'][cust-1]
            cust_addr = customer_addr_idx[cust-1]
            
            # Calculate savings of removing customer
            cost_pre = calculate_route_cost(src_route, customer_addr_idx, distance_matrix_array, depot_idx)
            temp_src = src_route[:i] + src_route[i+1:]
            cost_post = calculate_route_cost(temp_src, customer_addr_idx, distance_matrix_array, depot_idx)
            savings = cost_pre - cost_post
            
            # Try inserting into other routes
            for r_dst_idx, dst_route in enumerate(solution.routes[:-1]):
                if r_src_idx == r_dst_idx: continue
                
                # 1. Capacity Check
                dst_demand = sum(customer_arrays['demand'][c-1] for c in dst_route)
                if dst_demand + cust_demand > capacities[r_dst_idx]: continue
                
                # 2. Find best insertion
                best_delta = float('inf')
                best_pos = -1
                
                dst_addrs = [depot_idx] + [customer_addr_idx[c-1] for c in dst_route] + [depot_idx]
                
                for j in range(len(dst_route) + 1):
                    prev = dst_addrs[j]
                    next_node = dst_addrs[j+1]
                    
                    # Marginal cost of insertion
                    increase = (distance_matrix_array[prev, cust_addr] + 
                                distance_matrix_array[cust_addr, next_node] - 
                                distance_matrix_array[prev, next_node])
                    
                    # Net change
                    delta = increase - savings
                    
                    if delta < -1e-3: # Improvement found
                        if delta < best_delta:
                            # 3. Feasibility Check (Lazy)
                            candidate_dst = dst_route[:j] + [cust] + dst_route[j:]
                            if check_time_window_feasibility(candidate_dst, distance_matrix_array, customer_addr_idx, customer_arrays, depot_idx):
                                best_delta = delta
                                best_pos = j
                
                if best_pos != -1:
                    solution.routes[r_src_idx] = temp_src
                    solution.routes[r_dst_idx].insert(best_pos, cust)
                    return True # First improvement
                    
    return False

def cross_route_segment_relocation(solution, distance_matrix_array, customer_addr_idx, customer_arrays, vehicles_df, depot_idx=0):
    """
    HEAVY INTER-ROUTE: Relocates segments (chains of 1-3 customers).
    Computational expensive, run periodically.
    """
    improved = False
    capacities = [vehicles_df.loc[v, 'capacity'] for v in solution.vehicles[:-1]]
    
    # Try relocating segments of 1, 2, 3 customers
    for seg_len in [1, 2, 3]:
        for r_idx in range(len(solution.routes) - 1):
            route_src = solution.routes[r_idx]
            if len(route_src) < seg_len: continue
            
            for seg_start in range(len(route_src) - seg_len + 1):
                segment = route_src[seg_start:seg_start + seg_len]
                
                # Source route without segment
                new_src = route_src[:seg_start] + route_src[seg_start + seg_len:]
                src_cost_old = calculate_route_cost(route_src, customer_addr_idx, distance_matrix_array, depot_idx)
                
                for r_dst in range(len(solution.routes) - 1):
                    if r_dst == r_idx: continue
                    route_dst = solution.routes[r_dst]
                    
                    for pos in range(len(route_dst) + 1):
                        new_dst = route_dst[:pos] + segment + route_dst[pos:]
                        
                        # Capacity Check
                        src_demand = sum(customer_arrays['demand'][c-1] for c in new_src)
                        dst_demand = sum(customer_arrays['demand'][c-1] for c in new_dst)
                        if src_demand > capacities[r_idx] or dst_demand > capacities[r_dst]:
                            continue
                        
                        # Time Window Check
                        if not check_time_window_feasibility(new_src, distance_matrix_array, customer_addr_idx, customer_arrays, depot_idx): continue
                        if not check_time_window_feasibility(new_dst, distance_matrix_array, customer_addr_idx, customer_arrays, depot_idx): continue
                        
                        # Cost Check
                        src_cost_new = calculate_route_cost(new_src, customer_addr_idx, distance_matrix_array, depot_idx)
                        dst_cost_old = calculate_route_cost(route_dst, customer_addr_idx, distance_matrix_array, depot_idx)
                        dst_cost_new = calculate_route_cost(new_dst, customer_addr_idx, distance_matrix_array, depot_idx)
                        
                        delta = (src_cost_new - src_cost_old) + (dst_cost_new - dst_cost_old)
                        
                        if delta < -1e-3:
                            solution.routes[r_idx] = new_src
                            solution.routes[r_dst] = new_dst
                            return True
    return improved

# ---------------------------------------------------------
#  OPERATORS (Destroy)
# ---------------------------------------------------------

def random_removal(solution, num_to_remove, **kwargs):
    new_sol = solution.copy()
    candidates = []
    for r_idx, route in enumerate(new_sol.routes[:-1]):
        for cust in route:
            candidates.append(cust)
            
    if not candidates: return new_sol

    num_to_remove = min(len(candidates), num_to_remove)
    to_remove = set(random.sample(candidates, num_to_remove))
    
    for i in range(len(new_sol.routes) - 1):
        new_sol.routes[i] = [c for c in new_sol.routes[i] if c not in to_remove]
        
    new_sol.routes[-1].extend(list(to_remove))
    return new_sol

def worst_removal(solution, num_to_remove, distance_matrix_array, customer_addr_idx, depot_idx=0, **kwargs):
    new_sol = solution.copy()
    savings_list = [] 

    for r_idx, route in enumerate(new_sol.routes[:-1]):
        if not route: continue
        
        # Use delta cost instead of full recalculation
        for i, cust in enumerate(route):
            cust_addr = customer_addr_idx[cust-1]
            prev_addr = depot_idx if i == 0 else customer_addr_idx[route[i-1]-1]
            next_addr = depot_idx if i == len(route)-1 else customer_addr_idx[route[i+1]-1]
            
            # Cost saved if we remove this node
            removed_cost = distance_matrix_array[prev_addr, cust_addr] + distance_matrix_array[cust_addr, next_addr]
            added_cost = distance_matrix_array[prev_addr, next_addr]
            savings = removed_cost - added_cost
            
            savings_list.append((savings, cust))
    
    # Sort by highest savings (most expensive edges)
    savings_list.sort(key=lambda x: x[0], reverse=True)
    
    # Randomization to avoid determinism (Randomized Worst Removal)
    targets = set()
    while len(targets) < num_to_remove and savings_list:
        # Pick from top-k or using a power distribution
        idx = int(len(savings_list) * (random.random()**3)) # Skew towards index 0
        targets.add(savings_list.pop(idx)[1])
            
    for i in range(len(new_sol.routes) - 1):
        new_sol.routes[i] = [c for c in new_sol.routes[i] if c not in targets]
        
    new_sol.routes[-1].extend(list(targets))
    return new_sol

def cluster_removal(solution, num_to_remove, distance_matrix_array, customer_addr_idx, **kwargs):
    new_sol = solution.copy()
    candidates = []
    for r in new_sol.routes[:-1]:
        candidates.extend(r)
    
    if not candidates: return new_sol
    
    center_cust = random.choice(candidates)
    center_idx = customer_addr_idx[center_cust-1]
    
    distances = []
    for cust in candidates:
        if cust == center_cust: continue
        idx = customer_addr_idx[cust-1]
        dist = distance_matrix_array[center_idx, idx]
        distances.append((dist, cust))
    
    distances.sort(key=lambda x: x[0])
    
    targets = {center_cust}
    for _, cust in distances[:num_to_remove-1]:
        targets.add(cust)
    
    for i in range(len(new_sol.routes) - 1):
        new_sol.routes[i] = [c for c in new_sol.routes[i] if c not in targets]
        
    new_sol.routes[-1].extend(list(targets))
    return new_sol

def shaw_removal(solution, num_to_remove, distance_matrix_array, customer_addr_idx, customer_arrays, **kwargs):
    """
    Shaw Removal (Related Removal).
    Removes customers that are similar in Distance, Time, and Demand.
    """
    new_sol = solution.copy()
    
    # Gather all assigned customers
    candidates = []
    for r in new_sol.routes[:-1]:
        candidates.extend(r)
    
    if not candidates: return new_sol
    
    # Pick random seed
    seed = random.choice(candidates)
    removed = {seed}
    
    # Weights
    w_dist, w_time, w_dem = 9.0, 3.0, 2.0
    
    # Normalization factors (approx)
    max_dist = np.max(distance_matrix_array) if np.max(distance_matrix_array) > 0 else 1.0
    
    # FIX: Use 'tw_end' and 'demand' from your feasibility.py keys
    max_time = np.max(customer_arrays['tw_end']) if np.max(customer_arrays['tw_end']) > 0 else 1.0
    max_dem = np.max(customer_arrays['demand']) if np.max(customer_arrays['demand']) > 0 else 1.0

    while len(removed) < num_to_remove and len(removed) < len(candidates):
        # Pick a customer already removed to be the reference
        ref_cust = random.choice(list(removed))
        ref_idx = customer_addr_idx[ref_cust-1]
        
        # FIX: Use 'tw_start' for time comparison
        ref_tw = customer_arrays['tw_start'][ref_cust-1]
        ref_d = customer_arrays['demand'][ref_cust-1]
        
        scored_candidates = []
        
        # Scan a sample of candidates to avoid O(N^2)
        sample_pool = [c for c in candidates if c not in removed]
        if len(sample_pool) > 50:
            sample_pool = random.sample(sample_pool, 50)
            
        for cust in sample_pool:
            c_idx = customer_addr_idx[cust-1]
            dist_val = distance_matrix_array[ref_idx, c_idx] / max_dist
            
            # FIX: Use 'tw_start' for time comparison
            time_val = abs(ref_tw - customer_arrays['tw_start'][cust-1]) / max_time
            dem_val = abs(ref_d - customer_arrays['demand'][cust-1]) / max_dem
            
            relatedness = w_dist * dist_val + w_time * time_val + w_dem * dem_val
            scored_candidates.append((relatedness, cust))
        
        scored_candidates.sort(key=lambda x: x[0])
        
        if scored_candidates:
            # Pick one of the top 3 most related
            idx = random.randint(0, min(2, len(scored_candidates)-1))
            removed.add(scored_candidates[idx][1])
        else:
            break
            
    for i in range(len(new_sol.routes) - 1):
        new_sol.routes[i] = [c for c in new_sol.routes[i] if c not in removed]
        
    new_sol.routes[-1].extend(list(removed))
    return new_sol

# ---------------------------------------------------------
#  OPERATORS (Repair)
# ---------------------------------------------------------

def greedy_insertion(solution, distance_matrix_array, customer_addr_idx, customer_arrays, vehicles_df, neighbor_sets, depot_idx=0, **kwargs):
    new_sol = solution.copy()
    unassigned = list(new_sol.routes[-1])
    new_sol.routes[-1] = []
    random.shuffle(unassigned) # Randomize order to vary greedy results
    
    capacities = [vehicles_df.loc[v, 'capacity'] for v in new_sol.vehicles[:-1]]
    
    def softmax_costs(costs):
        # lower cost => higher probability
        costs = np.array(costs, dtype=np.float64)
        scores = -costs  # invert so lower cost gets larger score
        scores -= scores.max()  # stabilize
        exp = np.exp(scores)
        return exp / exp.sum()
    
    for cust in unassigned:
        cust_addr = customer_addr_idx[cust-1]
        cust_demand = customer_arrays['demand'][cust-1]
        allowed = neighbor_sets[cust_addr]
        
        feasible = []  # (delta, r_idx, i)
        
        for r_idx in range(len(new_sol.routes) - 1):
            route = new_sol.routes[r_idx]
            if sum(customer_arrays['demand'][c-1] for c in route) + cust_demand > capacities[r_idx]:
                continue
                
            route_addrs = [depot_idx] + [customer_addr_idx[c-1] for c in route] + [depot_idx]
            
            for i in range(len(route) + 1):
                prev = route_addrs[i]
                if prev != depot_idx and prev not in allowed:
                    continue

                next_node = route_addrs[i+1]
                delta = (distance_matrix_array[prev, cust_addr] + 
                         distance_matrix_array[cust_addr, next_node] - 
                         distance_matrix_array[prev, next_node])
                candidate = route[:i] + [cust] + route[i:]
                if check_time_window_feasibility(candidate, distance_matrix_array, customer_addr_idx, customer_arrays, depot_idx):
                    feasible.append((delta, r_idx, i))
        
        if feasible:
            deltas = [f[0] for f in feasible]
            probs = softmax_costs(deltas)
            choice_idx = np.random.choice(len(feasible), p=probs)
            _, r_idx, pos = feasible[choice_idx]
            new_sol.routes[r_idx].insert(pos, cust)
        else:
            new_sol.routes[-1].append(cust)
    return new_sol

def regret_insertion(solution, distance_matrix_array, customer_addr_idx, customer_arrays, vehicles_df, neighbor_sets, depot_idx=0, **kwargs):
    """2-Regret Insertion."""
    new_sol = solution.copy()
    unassigned = list(new_sol.routes[-1])
    new_sol.routes[-1] = []
    capacities = [vehicles_df.loc[v, 'capacity'] for v in new_sol.vehicles[:-1]]

    def softmax_costs(costs):
        costs = np.array(costs, dtype=np.float64)
        scores = -costs
        scores -= scores.max()
        exp = np.exp(scores)
        return exp / exp.sum()
    
    while unassigned:
        best_regret = -1
        best_cust = None
        best_pos = None
        found_any = False
        
        # To speed up, only evaluate a subset of unassigned if list is huge
        current_batch = unassigned if len(unassigned) < 30 else random.sample(unassigned, 30)
        
        for cust in current_batch:
            cust_addr = customer_addr_idx[cust-1]
            allowed = neighbor_sets[cust_addr]
            valid_insertions = []  # (cost, r_idx, i)
            
            for r_idx in range(len(new_sol.routes) - 1):
                route = new_sol.routes[r_idx]
                if sum(customer_arrays['demand'][c-1] for c in route) + customer_arrays['demand'][cust-1] > capacities[r_idx]:
                    continue
                
                route_addrs = [depot_idx] + [customer_addr_idx[c-1] for c in route] + [depot_idx]
                for i in range(len(route) + 1):
                    prev = route_addrs[i]
                    if prev != depot_idx and prev not in allowed: continue
                    
                    next_node = route_addrs[i+1]
                    cost = (distance_matrix_array[prev, cust_addr] + 
                            distance_matrix_array[cust_addr, next_node] - 
                            distance_matrix_array[prev, next_node])
                    valid_insertions.append((cost, r_idx, i))
            
            # Keep only feasible insertions
            feasible = []
            for cost, r, i in valid_insertions:
                cand = new_sol.routes[r][:i] + [cust] + new_sol.routes[r][i:]
                if check_time_window_feasibility(cand, distance_matrix_array, customer_addr_idx, customer_arrays, depot_idx):
                    feasible.append((cost, r, i))
            
            if not feasible:
                continue
            found_any = True
            
            # Probabilistic choice among feasible insertions (softmax on -cost)
            costs = [f[0] for f in feasible]
            probs = softmax_costs(costs)
            chosen_idx = np.random.choice(len(feasible), p=probs)
            chosen_cost, chosen_r, chosen_i = feasible[chosen_idx]
            
            # Regret based on best vs second-best costs (if available)
            feasible_sorted = sorted(feasible, key=lambda x: x[0])
            regret = feasible_sorted[1][0] - feasible_sorted[0][0] if len(feasible_sorted) > 1 else float('inf')
            
            if regret > best_regret:
                best_regret = regret
                best_cust = cust
                best_pos = (chosen_r, chosen_i)
        
        if not found_any:
            # Cannot insert any of the checked customers
            remaining = set(current_batch)
            for c in remaining:
                new_sol.routes[-1].append(c)
                unassigned.remove(c)
        else:
            new_sol.routes[best_pos[0]].insert(best_pos[1], best_cust)
            unassigned.remove(best_cust)
            
    return new_sol

# ---------------------------------------------------------
#  MAIN ALNS LOOP
# ---------------------------------------------------------

def create_initial_solution(num_customers, num_real_vehicles):
    routes = [[] for _ in range(num_real_vehicles)]
    routes.append(list(range(1, num_customers + 1)))
    vehicles = ['Standard'] * num_real_vehicles + [DUMMY_VEHICLE_NAME]
    return Solution(routes, vehicles)

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