import random
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feasibility import (
    check_time_window_feasibility, check_capacity_feasibility,
    check_vehicle_store_compatibility, check_shift_feasibility,
    check_lunch_break_feasibility, find_lunch_break_position,
    get_earliest_departure, LOADING_TIME, LUNCH_DURATION,
    validate_and_trim_routes,
)

# ---------------------------------------------------------
#  OPERATORS (Destroy) — no changes for multi-trip
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

def worst_removal(solution, num_to_remove, time_matrix_array, customer_addr_idx, depot_idx=0, **kwargs):
    new_sol = solution.copy()
    savings_list = []

    for r_idx, route in enumerate(new_sol.routes[:-1]):
        if not route: continue

        for i, cust in enumerate(route):
            cust_addr = customer_addr_idx[cust-1]
            prev_addr = depot_idx if i == 0 else customer_addr_idx[route[i-1]-1]
            next_addr = depot_idx if i == len(route)-1 else customer_addr_idx[route[i+1]-1]

            removed_cost = time_matrix_array[prev_addr, cust_addr] + time_matrix_array[cust_addr, next_addr]
            added_cost = time_matrix_array[prev_addr, next_addr]
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

def cluster_removal(solution, num_to_remove, time_matrix_array, customer_addr_idx, **kwargs):
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
        dist = time_matrix_array[center_idx, idx]
        distances.append((dist, cust))

    distances.sort(key=lambda x: x[0])

    targets = {center_cust}
    for _, cust in distances[:num_to_remove-1]:
        targets.add(cust)

    for i in range(len(new_sol.routes) - 1):
        new_sol.routes[i] = [c for c in new_sol.routes[i] if c not in targets]

    new_sol.routes[-1].extend(list(targets))
    return new_sol

def shaw_removal(solution, num_to_remove, time_matrix_array, customer_addr_idx, customer_arrays, **kwargs):
    new_sol = solution.copy()
    candidates = []
    for r in new_sol.routes[:-1]:
        candidates.extend(r)

    if not candidates: return new_sol

    seed = random.choice(candidates)
    removed = {seed}

    w_dist, w_time, w_dem = 9.0, 3.0, 2.0

    max_dist = np.max(time_matrix_array) if np.max(time_matrix_array) > 0 else 1.0
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
            dist_val = time_matrix_array[ref_idx, c_idx] / max_dist
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
#  Trip-2 validation (trip-1 changes can invalidate trip-2)
# ---------------------------------------------------------

def _validate_trip2_routes(solution, time_matrix_array, customer_addr_idx,
                           customer_arrays, depot_idx):
    """
    After repair, trip-1 may have grown, pushing trip-2 departure later.
    Check all trip-2 routes and move infeasible ones back to dummy.
    """
    for r_idx in range(len(solution.routes) - 1):
        meta = solution.route_meta[r_idx]
        if meta is None or meta['trip'] != 2 or not solution.routes[r_idx]:
            continue

        earliest_dep = get_earliest_departure(
            solution, r_idx, time_matrix_array, customer_addr_idx,
            customer_arrays, depot_idx
        )

        # Check if trip-2 still fits in shift
        feasible = check_shift_feasibility(
            solution.routes[r_idx], time_matrix_array, customer_addr_idx,
            customer_arrays, depot_idx, earliest_dep, meta['shift_end']
        )

        # Also check time windows with updated departure
        if feasible:
            feasible = check_time_window_feasibility(
                solution.routes[r_idx], time_matrix_array, customer_addr_idx,
                customer_arrays, depot_idx, earliest_departure=earliest_dep
            )

        if not feasible:
            solution.routes[-1].extend(solution.routes[r_idx])
            solution.routes[r_idx] = []
            solution.lunch_breaks[r_idx] = None


# ---------------------------------------------------------
#  OPERATORS (Repair) — shift-aware for multi-trip
# ---------------------------------------------------------

def greedy_insertion(solution, time_matrix_array, customer_addr_idx, customer_arrays, vehicles_dict, neighbor_sets, depot_idx=0, temperature=1.0, **kwargs):
    """Greedy insertion with two-phase approach: trip-1 first, then trip-2."""
    new_sol = solution.copy()
    unassigned = list(new_sol.routes[-1])
    new_sol.routes[-1] = []
    random.shuffle(unassigned)
    compatible_ppls_set = kwargs.get('compatible_ppls_set', set())

    for phase_trip in (1, 2):
        # Between phases: validate existing trip-2 routes against updated trip-1
        if phase_trip == 2:
            _validate_trip2_routes(new_sol, time_matrix_array, customer_addr_idx,
                                   customer_arrays, depot_idx)
            unassigned.extend(new_sol.routes[-1])
            new_sol.routes[-1] = []
            random.shuffle(unassigned)

        still_unassigned = []
        for cust in unassigned:
            cust_addr = customer_addr_idx[cust-1]
            feasible_points = []

            for r_idx in range(len(new_sol.routes) - 1):
                route = new_sol.routes[r_idx]
                vehicle_name = new_sol.vehicles[r_idx]
                meta = new_sol.route_meta[r_idx]

                # Phase filter: only consider slots matching current phase
                if meta is None or meta['trip'] != phase_trip:
                    continue
                # Skip trip-2 when trip-1 in same shift is empty
                if phase_trip == 2 and not new_sol.routes[r_idx - 1]:
                    continue

                if not check_capacity_feasibility(route + [cust], vehicle_name, vehicles_dict, customer_arrays):
                    continue
                if not check_vehicle_store_compatibility(route + [cust], vehicle_name, vehicles_dict, customer_arrays, compatible_ppls_set):
                    continue

                earliest_dep = get_earliest_departure(
                    new_sol, r_idx, time_matrix_array, customer_addr_idx,
                    customer_arrays, depot_idx
                )
                shift_end = meta['shift_end']

                route_addrs = [depot_idx] + [customer_addr_idx[c-1] for c in route] + [depot_idx]
                for i in range(len(route) + 1):
                    prev, nxt = route_addrs[i], route_addrs[i+1]
                    delta = (time_matrix_array[prev, cust_addr] +
                             time_matrix_array[cust_addr, nxt] -
                             time_matrix_array[prev, nxt])
                    candidate_route = route[:i] + [cust] + route[i:]
                    if not check_time_window_feasibility(candidate_route, time_matrix_array, customer_addr_idx, customer_arrays, depot_idx, earliest_departure=earliest_dep):
                        continue
                    lunch_buf = LUNCH_DURATION if phase_trip == 1 else 0.0
                    if not check_shift_feasibility(candidate_route, time_matrix_array, customer_addr_idx, customer_arrays, depot_idx, earliest_dep, shift_end, lunch_duration=lunch_buf):
                        continue
                    if phase_trip == 1:
                        if not check_lunch_break_feasibility(candidate_route, time_matrix_array, customer_addr_idx, customer_arrays, depot_idx,
                                                             earliest_departure=earliest_dep, shift_start=meta['shift_start']):
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
                sel_meta = new_sol.route_meta[selected_r]
                if phase_trip == 1:
                    sel_dep = get_earliest_departure(new_sol, selected_r, time_matrix_array, customer_addr_idx, customer_arrays, depot_idx)
                    new_sol.lunch_breaks[selected_r] = find_lunch_break_position(
                        new_sol.routes[selected_r], time_matrix_array, customer_addr_idx,
                        customer_arrays, depot_idx, earliest_departure=sel_dep,
                        shift_start=sel_meta['shift_start'])
                else:
                    new_sol.lunch_breaks[selected_r] = None
            else:
                still_unassigned.append(cust)

        unassigned = still_unassigned

    new_sol.routes[-1] = unassigned
    # Safety net: trim any routes that overflow the shift
    validate_and_trim_routes(new_sol, time_matrix_array, customer_addr_idx,
                             customer_arrays, depot_idx)
    return new_sol

# ---------------------------------------------------------
#  LOCAL SEARCH — 2-opt (shift-aware)
# ---------------------------------------------------------

def two_opt_local_search(solution, time_matrix_array, customer_addr_idx, customer_arrays, depot_idx=0, **kwargs):
    """
    Apply 2-opt local search to every route (multi-trip aware).
    """
    new_sol = solution.copy()

    for r_idx in range(len(new_sol.routes) - 1):
        route = new_sol.routes[r_idx]
        n = len(route)
        if n < 2:
            continue

        meta = new_sol.route_meta[r_idx]
        earliest_dep = get_earliest_departure(
            new_sol, r_idx, time_matrix_array, customer_addr_idx,
            customer_arrays, depot_idx
        )
        shift_end = meta['shift_end'] if meta else 22.0
        shift_start = meta['shift_start'] if meta else 6.0
        is_trip1 = meta is not None and meta['trip'] == 1

        route_addrs = [depot_idx] + [customer_addr_idx[c - 1] for c in route] + [depot_idx]

        improved = True
        passes = 0
        while improved and passes < 10:
            improved = False
            passes += 1
            for i in range(1, n + 1):
                for j in range(i + 1, n + 1):
                    old_edges = (time_matrix_array[route_addrs[i - 1], route_addrs[i]] +
                                 time_matrix_array[route_addrs[j], route_addrs[j + 1]])
                    new_edges = (time_matrix_array[route_addrs[i - 1], route_addrs[j]] +
                                 time_matrix_array[route_addrs[i], route_addrs[j + 1]])

                    if new_edges < old_edges - 1e-6:
                        candidate = route[:i - 1] + list(reversed(route[i - 1:j])) + route[j:]
                        if not check_time_window_feasibility(candidate, time_matrix_array, customer_addr_idx, customer_arrays, depot_idx, earliest_departure=earliest_dep):
                            continue
                        lunch_buf = LUNCH_DURATION if is_trip1 else 0.0
                        if not check_shift_feasibility(candidate, time_matrix_array, customer_addr_idx, customer_arrays, depot_idx, earliest_dep, shift_end, lunch_duration=lunch_buf):
                            continue
                        if is_trip1:
                            if not check_lunch_break_feasibility(candidate, time_matrix_array, customer_addr_idx, customer_arrays, depot_idx,
                                                                  earliest_departure=earliest_dep, shift_start=shift_start):
                                continue
                        route = candidate
                        route_addrs = [depot_idx] + [customer_addr_idx[c - 1] for c in route] + [depot_idx]
                        improved = True
                        break
                if improved:
                    break

        new_sol.routes[r_idx] = route
        if is_trip1:
            new_sol.lunch_breaks[r_idx] = find_lunch_break_position(
                route, time_matrix_array, customer_addr_idx, customer_arrays, depot_idx,
                earliest_departure=earliest_dep, shift_start=shift_start)
        else:
            new_sol.lunch_breaks[r_idx] = None

    return new_sol


def or_opt_local_search(solution, time_matrix_array, customer_addr_idx, customer_arrays, depot_idx=0, **kwargs):
    """
    Apply or-opt-1 (relocate) local search to every route (multi-trip aware).
    """
    new_sol = solution.copy()

    for r_idx in range(len(new_sol.routes) - 1):
        route = new_sol.routes[r_idx]
        if len(route) < 2:
            continue

        meta = new_sol.route_meta[r_idx]
        earliest_dep = get_earliest_departure(
            new_sol, r_idx, time_matrix_array, customer_addr_idx,
            customer_arrays, depot_idx
        )
        shift_end = meta['shift_end'] if meta else 22.0
        shift_start = meta['shift_start'] if meta else 6.0
        is_trip1 = meta is not None and meta['trip'] == 1

        improved = True
        passes = 0
        while improved and passes < 10:
            improved = False
            passes += 1
            n = len(route)
            route_addrs = [depot_idx] + [customer_addr_idx[c - 1] for c in route] + [depot_idx]

            for k in range(1, n + 1):  # 1-based position of customer to relocate
                prev_k = route_addrs[k - 1]
                curr_k = route_addrs[k]
                next_k = route_addrs[k + 1]
                removal_gain = (time_matrix_array[prev_k, curr_k] +
                                time_matrix_array[curr_k, next_k] -
                                time_matrix_array[prev_k, next_k])

                # Reduced address list after removing customer k
                reduced_addrs = route_addrs[:k] + route_addrs[k + 1:]

                for p in range(1, n + 1):  # 1-based insertion position in reduced route
                    if p == k:  # same position — skip
                        continue
                    prev_p = reduced_addrs[p - 1]
                    next_p = reduced_addrs[p]
                    insertion_cost = (time_matrix_array[prev_p, curr_k] +
                                      time_matrix_array[curr_k, next_p] -
                                      time_matrix_array[prev_p, next_p])

                    if removal_gain - insertion_cost > 1e-6:
                        route_without_k = route[:k - 1] + route[k:]
                        candidate = route_without_k[:p - 1] + [route[k - 1]] + route_without_k[p - 1:]
                        if not check_time_window_feasibility(candidate, time_matrix_array, customer_addr_idx, customer_arrays, depot_idx, earliest_departure=earliest_dep):
                            continue
                        lunch_buf = LUNCH_DURATION if is_trip1 else 0.0
                        if not check_shift_feasibility(candidate, time_matrix_array, customer_addr_idx, customer_arrays, depot_idx, earliest_dep, shift_end, lunch_duration=lunch_buf):
                            continue
                        if is_trip1:
                            if not check_lunch_break_feasibility(candidate, time_matrix_array, customer_addr_idx, customer_arrays, depot_idx,
                                                                  earliest_departure=earliest_dep, shift_start=shift_start):
                                continue
                        route = candidate
                        improved = True
                        break
                if improved:
                    break

        new_sol.routes[r_idx] = route
        if is_trip1:
            new_sol.lunch_breaks[r_idx] = find_lunch_break_position(
                route, time_matrix_array, customer_addr_idx, customer_arrays, depot_idx,
                earliest_departure=earliest_dep, shift_start=shift_start)
        else:
            new_sol.lunch_breaks[r_idx] = None

    return new_sol


def regret_insertion(solution, time_matrix_array, customer_addr_idx, customer_arrays, vehicles_dict, neighbor_sets, depot_idx=0, temperature=1.0, **kwargs):
    """2-Regret insertion with two-phase approach: trip-1 first, then trip-2."""
    new_sol = solution.copy()
    unassigned = list(new_sol.routes[-1])
    new_sol.routes[-1] = []
    compatible_ppls_set = kwargs.get('compatible_ppls_set', set())

    for phase_trip in (1, 2):
        # Between phases: validate existing trip-2 routes against updated trip-1
        if phase_trip == 2:
            _validate_trip2_routes(new_sol, time_matrix_array, customer_addr_idx,
                                   customer_arrays, depot_idx)
            unassigned.extend(new_sol.routes[-1])
            new_sol.routes[-1] = []

        while unassigned:
            potential_insertions = []

            for cust in unassigned:
                cust_addr = customer_addr_idx[cust-1]
                feasible_options = []

                for r_idx in range(len(new_sol.routes) - 1):
                    route = new_sol.routes[r_idx]
                    vehicle_name = new_sol.vehicles[r_idx]
                    meta = new_sol.route_meta[r_idx]

                    # Phase filter: only consider slots matching current phase
                    if meta is None or meta['trip'] != phase_trip:
                        continue
                    # Skip trip-2 when trip-1 in same shift is empty
                    if phase_trip == 2 and not new_sol.routes[r_idx - 1]:
                        continue

                    if not check_capacity_feasibility(route + [cust], vehicle_name, vehicles_dict, customer_arrays):
                        continue
                    if not check_vehicle_store_compatibility(route + [cust], vehicle_name, vehicles_dict, customer_arrays, compatible_ppls_set):
                        continue

                    earliest_dep = get_earliest_departure(
                        new_sol, r_idx, time_matrix_array, customer_addr_idx,
                        customer_arrays, depot_idx
                    )
                    shift_end = meta['shift_end']

                    route_addrs = [depot_idx] + [customer_addr_idx[c-1] for c in route] + [depot_idx]
                    for i in range(len(route) + 1):
                        prev, nxt = route_addrs[i], route_addrs[i+1]
                        delta = (time_matrix_array[prev, cust_addr] +
                                 time_matrix_array[cust_addr, nxt] -
                                 time_matrix_array[prev, nxt])
                        temp_route = route[:i] + [cust] + route[i:]
                        if not check_time_window_feasibility(temp_route, time_matrix_array, customer_addr_idx, customer_arrays, depot_idx, earliest_departure=earliest_dep):
                            continue
                        lunch_buf = LUNCH_DURATION if phase_trip == 1 else 0.0
                        if not check_shift_feasibility(temp_route, time_matrix_array, customer_addr_idx, customer_arrays, depot_idx, earliest_dep, shift_end, lunch_duration=lunch_buf):
                            continue
                        if phase_trip == 1:
                            if not check_lunch_break_feasibility(temp_route, time_matrix_array, customer_addr_idx, customer_arrays, depot_idx,
                                                                 earliest_departure=earliest_dep, shift_start=meta['shift_start']):
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
                break  # No more insertions possible for this phase

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
            sel_meta = new_sol.route_meta[final_r]
            if phase_trip == 1:
                sel_dep = get_earliest_departure(new_sol, final_r, time_matrix_array, customer_addr_idx, customer_arrays, depot_idx)
                new_sol.lunch_breaks[final_r] = find_lunch_break_position(
                    new_sol.routes[final_r], time_matrix_array, customer_addr_idx,
                    customer_arrays, depot_idx, earliest_departure=sel_dep,
                    shift_start=sel_meta['shift_start'])
            else:
                new_sol.lunch_breaks[final_r] = None
            unassigned.remove(target_cust)

    new_sol.routes[-1] = unassigned
    # Safety net: trim any routes that overflow the shift
    validate_and_trim_routes(new_sol, time_matrix_array, customer_addr_idx,
                             customer_arrays, depot_idx)
    return new_sol
