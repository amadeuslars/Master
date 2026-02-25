import random
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feasibility import check_time_window_feasibility, check_capacity_feasibility, check_vehicle_store_compatibility, check_max_route_duration, check_lunch_break_feasibility, find_lunch_break_position

# ---------------------------------------------------------
#  OPERATORS (Destroy) — identical to benchmark
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
#  OPERATORS (Repair) — benchmark logic + case study feasibility
# ---------------------------------------------------------

def greedy_insertion(solution, distance_matrix_array, customer_addr_idx, customer_arrays, vehicles_dict, neighbor_sets, depot_idx=0, temperature=1.0, **kwargs):
    """Greedy insertion with Blended Softmax selection."""
    new_sol = solution.copy()
    unassigned = list(new_sol.routes[-1])
    new_sol.routes[-1] = []
    random.shuffle(unassigned)
    compatible_ppls_set = kwargs.get('compatible_ppls_set', set())

    for cust in unassigned:
        cust_addr = customer_addr_idx[cust-1]
        feasible_points = []

        for r_idx in range(len(new_sol.routes) - 1):
            route = new_sol.routes[r_idx]
            vehicle_name = new_sol.vehicles[r_idx]

            if not check_capacity_feasibility(route + [cust], vehicle_name, vehicles_dict, customer_arrays):
                continue
            if not check_vehicle_store_compatibility(route + [cust], vehicle_name, vehicles_dict, customer_arrays, compatible_ppls_set):
                continue

            route_addrs = [depot_idx] + [customer_addr_idx[c-1] for c in route] + [depot_idx]
            for i in range(len(route) + 1):
                prev, nxt = route_addrs[i], route_addrs[i+1]
                delta = (distance_matrix_array[prev, cust_addr] +
                         distance_matrix_array[cust_addr, nxt] -
                         distance_matrix_array[prev, nxt])
                candidate_route = route[:i] + [cust] + route[i:]
                if not check_time_window_feasibility(candidate_route, distance_matrix_array, customer_addr_idx, customer_arrays, depot_idx):
                    continue
                if not check_max_route_duration(candidate_route, customer_addr_idx, distance_matrix_array, depot_idx):
                    continue
                if not check_lunch_break_feasibility(candidate_route, distance_matrix_array, customer_addr_idx, customer_arrays, depot_idx):
                    continue
                feasible_points.append((delta, r_idx, i))

        if feasible_points:
            costs = np.array([p[0] for p in feasible_points], dtype=np.float64)
            norm_costs = (costs - np.min(costs)) / (temperature + 1e-9)
            exp_neg_costs = np.exp(-norm_costs)
            probs = exp_neg_costs / np.sum(exp_neg_costs)
            idx = np.random.choice(len(feasible_points), p=probs)
            _, selected_r, selected_p = feasible_points[idx]
            new_sol.routes[selected_r].insert(selected_p, cust)
            new_sol.lunch_breaks[selected_r] = find_lunch_break_position(
                new_sol.routes[selected_r], distance_matrix_array, customer_addr_idx,
                customer_arrays, depot_idx)
        else:
            new_sol.routes[-1].append(cust)
    return new_sol

def regret_insertion(solution, distance_matrix_array, customer_addr_idx, customer_arrays, vehicles_dict, neighbor_sets, depot_idx=0, temperature=1.0, **kwargs):
    """2-Regret insertion with Blended Softmax for position selection."""
    new_sol = solution.copy()
    unassigned = list(new_sol.routes[-1])
    new_sol.routes[-1] = []
    compatible_ppls_set = kwargs.get('compatible_ppls_set', set())

    while unassigned:
        potential_insertions = []

        for cust in unassigned:
            cust_addr = customer_addr_idx[cust-1]
            feasible_options = []

            for r_idx in range(len(new_sol.routes) - 1):
                route = new_sol.routes[r_idx]
                vehicle_name = new_sol.vehicles[r_idx]

                if not check_capacity_feasibility(route + [cust], vehicle_name, vehicles_dict, customer_arrays):
                    continue
                if not check_vehicle_store_compatibility(route + [cust], vehicle_name, vehicles_dict, customer_arrays, compatible_ppls_set):
                    continue

                route_addrs = [depot_idx] + [customer_addr_idx[c-1] for c in route] + [depot_idx]
                for i in range(len(route) + 1):
                    prev, nxt = route_addrs[i], route_addrs[i+1]
                    delta = (distance_matrix_array[prev, cust_addr] +
                             distance_matrix_array[cust_addr, nxt] -
                             distance_matrix_array[prev, nxt])
                    temp_route = route[:i] + [cust] + route[i:]
                    if not check_time_window_feasibility(temp_route, distance_matrix_array, customer_addr_idx, customer_arrays, depot_idx):
                        continue
                    if not check_max_route_duration(temp_route, customer_addr_idx, distance_matrix_array, depot_idx):
                        continue
                    if not check_lunch_break_feasibility(temp_route, distance_matrix_array, customer_addr_idx, customer_arrays, depot_idx):
                        continue
                    feasible_options.append((delta, r_idx, i))

            if not feasible_options:
                continue

            feasible_options.sort(key=lambda x: x[0])

            if len(feasible_options) >= 2:
                regret_val = feasible_options[1][0] - feasible_options[0][0]
            else:
                regret_val = 1e6

            potential_insertions.append((regret_val, cust, feasible_options))

        if not potential_insertions:
            new_sol.routes[-1].extend(unassigned)
            break

        potential_insertions.sort(key=lambda x: x[0], reverse=True)
        best_regret_match = potential_insertions[0]

        target_cust = best_regret_match[1]
        available_options = best_regret_match[2]

        costs = np.array([opt[0] for opt in available_options])
        norm_costs = (costs - np.min(costs)) / (temperature + 1e-9)
        probs = np.exp(-norm_costs) / np.sum(np.exp(-norm_costs))

        choice_idx = np.random.choice(len(available_options), p=probs)
        _, final_r, final_p = available_options[choice_idx]

        new_sol.routes[final_r].insert(final_p, target_cust)
        new_sol.lunch_breaks[final_r] = find_lunch_break_position(
            new_sol.routes[final_r], distance_matrix_array, customer_addr_idx,
            customer_arrays, depot_idx)
        unassigned.remove(target_cust)

    return new_sol
