import random
import numpy as np
from utils.feasibility import check_capacity_feasibility, check_time_window_feasibility

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
            
            removed_cost = distance_matrix_array[prev_addr, cust_addr] + distance_matrix_array[cust_addr, next_addr]
            added_cost = distance_matrix_array[prev_addr, next_addr]
            savings = removed_cost - added_cost
            
            savings_list.append((savings, cust))
    
    savings_list.sort(key=lambda x: x[0], reverse=True)
    
    targets = set()
    while len(targets) < num_to_remove and savings_list:
        idx = int(len(savings_list) * (random.random()**3)) 
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
    new_sol = solution.copy()
    candidates = []
    for r in new_sol.routes[:-1]:
        candidates.extend(r)
    
    if not candidates: return new_sol
    
    seed = random.choice(candidates)
    removed = {seed}
    
    w_dist, w_time, w_dem = 9.0, 3.0, 2.0
    
    max_dist = np.max(distance_matrix_array) if np.max(distance_matrix_array) > 0 else 1.0
    max_time = np.max(customer_arrays['tw_end']) if np.max(customer_arrays['tw_end']) > 0 else 1.0
    max_dem = np.max(customer_arrays['demand']) if np.max(customer_arrays['demand']) > 0 else 1.0

    while len(removed) < num_to_remove and len(removed) < len(candidates):
        ref_cust = random.choice(list(removed))
        ref_idx = customer_addr_idx[ref_cust-1]
        
        ref_tw = customer_arrays['tw_start'][ref_cust-1]
        ref_d = customer_arrays['demand'][ref_cust-1]
        
        scored_candidates = []
        sample_pool = [c for c in candidates if c not in removed]
        if len(sample_pool) > 50:
            sample_pool = random.sample(sample_pool, 50)
            
        for cust in sample_pool:
            c_idx = customer_addr_idx[cust-1]
            dist_val = distance_matrix_array[ref_idx, c_idx] / max_dist
            
            time_val = abs(ref_tw - customer_arrays['tw_start'][cust-1]) / max_time
            dem_val = abs(ref_d - customer_arrays['demand'][cust-1]) / max_dem
            
            relatedness = w_dist * dist_val + w_time * time_val + w_dem * dem_val
            scored_candidates.append((relatedness, cust))
        
        scored_candidates.sort(key=lambda x: x[0])
        
        if scored_candidates:
            idx = random.randint(0, min(2, len(scored_candidates)-1))
            removed.add(scored_candidates[idx][1])
        else:
            break
            
    for i in range(len(new_sol.routes) - 1):
        new_sol.routes[i] = [c for c in new_sol.routes[i] if c not in removed]
        
    new_sol.routes[-1].extend(list(removed))
    return new_sol

def least_used_vehicle_removal(solution, num_to_remove, **kwargs):
    """Prioritizes emptying the vehicles with the fewest customers."""
    new_sol = solution.copy()
    
    routes_info = []
    for i, route in enumerate(new_sol.routes[:-1]):
        if len(route) > 0:
            routes_info.append((i, len(route)))
    
    random.shuffle(routes_info)
    routes_info.sort(key=lambda x: x[1])
    
    targets = set()
    current_removed = 0
    
    for r_idx, r_len in routes_info:
        if current_removed + r_len <= num_to_remove:
            targets.update(new_sol.routes[r_idx])
            current_removed += r_len
        elif current_removed == 0:
            subset = random.sample(new_sol.routes[r_idx], num_to_remove)
            targets.update(subset)
            break
        else:
            break
            
    for i in range(len(new_sol.routes) - 1):
        new_sol.routes[i] = [c for c in new_sol.routes[i] if c not in targets]
        
    new_sol.routes[-1].extend(list(targets))
    return new_sol

# ---------------------------------------------------------
#  OPERATORS (Repair)
# ---------------------------------------------------------

def greedy_insertion(solution, distance_matrix_array, customer_addr_idx, customer_arrays, vehicles_df, neighbor_sets, depot_idx=0, **kwargs):
    new_sol = solution.copy()
    unassigned = list(new_sol.routes[-1])
    new_sol.routes[-1] = []
    random.shuffle(unassigned) 
    
    capacities = [vehicles_df.loc[v, 'capacity'] for v in new_sol.vehicles[:-1]]
    
    def softmax_costs(costs):
        costs = np.array(costs, dtype=np.float64)
        scores = -costs 
        scores -= scores.max()
        exp = np.exp(scores)
        return exp / exp.sum()
    
    for cust in unassigned:
        cust_addr = customer_addr_idx[cust-1]
        cust_demand = customer_arrays['demand'][cust-1]
        allowed = neighbor_sets[cust_addr]
        
        feasible = []  
        
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
        
        current_batch = unassigned if len(unassigned) < 30 else random.sample(unassigned, 30)
        
        for cust in current_batch:
            cust_addr = customer_addr_idx[cust-1]
            allowed = neighbor_sets[cust_addr]
            valid_insertions = []  
            
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
            
            feasible = []
            for cost, r, i in valid_insertions:
                cand = new_sol.routes[r][:i] + [cust] + new_sol.routes[r][i:]
                if check_time_window_feasibility(cand, distance_matrix_array, customer_addr_idx, customer_arrays, depot_idx):
                    feasible.append((cost, r, i))
            
            if not feasible:
                continue
            found_any = True
            
            costs = [f[0] for f in feasible]
            probs = softmax_costs(costs)
            chosen_idx = np.random.choice(len(feasible), p=probs)
            chosen_cost, chosen_r, chosen_i = feasible[chosen_idx]
            
            feasible_sorted = sorted(feasible, key=lambda x: x[0])
            regret = feasible_sorted[1][0] - feasible_sorted[0][0] if len(feasible_sorted) > 1 else float('inf')
            
            if regret > best_regret:
                best_regret = regret
                best_cust = cust
                best_pos = (chosen_r, chosen_i)
        
        if not found_any:
            remaining = set(current_batch)
            for c in remaining:
                new_sol.routes[-1].append(c)
                unassigned.remove(c)
        else:
            new_sol.routes[best_pos[0]].insert(best_pos[1], best_cust)
            unassigned.remove(best_cust)
            
    return new_sol