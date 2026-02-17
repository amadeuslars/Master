# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# NOTE: To debug segfaults, temporarily set boundscheck=True and wraparound=True

"""
Cythonized ALNS Destroy & Repair Operators for VRPTW.

v2 — feasibility checks are now fully inlined as cdef functions.
     No more Python function calls or list construction in the hot loop.
"""

import random as py_random
import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.math cimport exp, fabs
from libc.time cimport time as c_time

# ── Seed the C RNG once at import time ──────────────────────
srand(<unsigned int>c_time(NULL))

# ── Type aliases ────────────────────────────────────────────
ctypedef np.float64_t DTYPE_f
ctypedef np.intp_t    DTYPE_i


# =============================================================
#              FAST C-LEVEL HELPERS
# =============================================================

cdef inline double c_rand_double() noexcept nogil:
    return rand() / (<double>RAND_MAX + 1.0)

cdef inline int c_rand_int(int upper) noexcept nogil:
    if upper <= 0:
        return 0
    return rand() % upper

cdef inline double delta_cost(double[:, :] dm,
                              int prev_addr, int cust_addr, int next_addr) noexcept nogil:
    return dm[prev_addr, cust_addr] + dm[cust_addr, next_addr] - dm[prev_addr, next_addr]


# =============================================================
#      INLINED FEASIBILITY CHECKS  (the key speedup)
# =============================================================

cdef bint c_check_tw_with_insertion(list route, int route_len,
                                     int insert_pos, int new_cust,
                                     double[:, :] dm,
                                     np.intp_t[:] addr_idx,
                                     double[:] tw_start,
                                     double[:] tw_end,
                                     double[:] service_time,
                                     int depot_idx):
    """
    Check time-window feasibility for route WITH a customer inserted at insert_pos,
    WITHOUT building a new Python list.

    Simulates visiting: route[0..insert_pos-1], new_cust, route[insert_pos..end]
    """
    cdef double current_time = 0.0
    cdef int current_loc = depot_idx
    cdef int i, cust, cust_addr
    cdef double travel, arrival
    cdef int total_len = route_len + 1

    for i in range(total_len):
        if i < insert_pos:
            cust = <int>route[i]
        elif i == insert_pos:
            cust = new_cust
        else:
            cust = <int>route[i - 1]

        cust_addr = addr_idx[cust - 1]
        travel = dm[current_loc, cust_addr]
        arrival = current_time + travel

        if arrival > tw_end[cust - 1]:
            return False

        if arrival < tw_start[cust - 1]:
            arrival = tw_start[cust - 1]

        current_time = arrival + service_time[cust - 1]
        current_loc = cust_addr

    return True


# =============================================================
#                     DESTROY  OPERATORS
# =============================================================

def random_removal(solution, int num_to_remove, **kwargs):
    new_sol = solution.copy()
    cdef list candidates = []
    cdef int r_idx
    cdef list route

    for r_idx in range(len(new_sol.routes) - 1):
        route = new_sol.routes[r_idx]
        candidates.extend(route)

    if not candidates:
        return new_sol

    if num_to_remove > len(candidates):
        num_to_remove = len(candidates)

    cdef set to_remove = set(py_random.sample(candidates, num_to_remove))
    cdef int i

    for i in range(len(new_sol.routes) - 1):
        new_sol.routes[i] = [c for c in new_sol.routes[i] if c not in to_remove]

    new_sol.routes[len(new_sol.routes)-1].extend(list(to_remove))
    return new_sol


def worst_removal(solution, int num_to_remove,
                  double[:, :] distance_matrix_array,
                  np.intp_t[:] customer_addr_idx,
                  int depot_idx=0, **kwargs):
    new_sol = solution.copy()

    cdef list savings_list = []
    cdef int r_idx, i, cust, route_len
    cdef int cust_addr, prev_addr, next_addr
    cdef double removed_cost, added_cost, saving
    cdef list route

    for r_idx in range(len(new_sol.routes) - 1):
        route = new_sol.routes[r_idx]
        route_len = len(route)
        if route_len == 0:
            continue

        for i in range(route_len):
            cust = route[i]
            cust_addr = customer_addr_idx[cust - 1]
            prev_addr = depot_idx if i == 0 else customer_addr_idx[route[i - 1] - 1]
            next_addr = depot_idx if i == route_len - 1 else customer_addr_idx[route[i + 1] - 1]

            removed_cost = distance_matrix_array[prev_addr, cust_addr] + \
                           distance_matrix_array[cust_addr, next_addr]
            added_cost   = distance_matrix_array[prev_addr, next_addr]
            saving = removed_cost - added_cost

            savings_list.append((saving, cust))

    savings_list.sort(key=lambda x: x[0], reverse=True)

    cdef set targets = set()
    cdef int idx
    cdef double r

    while len(targets) < <unsigned int>num_to_remove and savings_list:
        r = c_rand_double()
        idx = <int>(len(savings_list) * (r * r * r))
        targets.add(savings_list.pop(idx)[1])

    for i in range(len(new_sol.routes) - 1):
        new_sol.routes[i] = [c for c in new_sol.routes[i] if c not in targets]

    new_sol.routes[len(new_sol.routes)-1].extend(list(targets))
    return new_sol


def cluster_removal(solution, int num_to_remove,
                    double[:, :] distance_matrix_array,
                    np.intp_t[:] customer_addr_idx, **kwargs):
    new_sol = solution.copy()

    cdef list candidates = []
    cdef list route
    for route in new_sol.routes[:-1]:
        candidates.extend(route)

    if not candidates:
        return new_sol

    cdef int center_cust = candidates[c_rand_int(len(candidates))]
    cdef int center_idx  = customer_addr_idx[center_cust - 1]

    cdef int n_cand = len(candidates)
    cdef list distances = []
    cdef int cust, c_idx
    cdef double dist

    for cust in candidates:
        if cust == center_cust:
            continue
        c_idx = customer_addr_idx[cust - 1]
        dist  = distance_matrix_array[center_idx, c_idx]
        distances.append((dist, cust))

    distances.sort(key=lambda x: x[0])

    cdef set targets = {center_cust}
    cdef int j
    for j in range(min(num_to_remove - 1, len(distances))):
        targets.add(distances[j][1])

    cdef int i
    for i in range(len(new_sol.routes) - 1):
        new_sol.routes[i] = [c for c in new_sol.routes[i] if c not in targets]

    new_sol.routes[len(new_sol.routes)-1].extend(list(targets))
    return new_sol


def shaw_removal(solution, int num_to_remove,
                 double[:, :] distance_matrix_array,
                 np.intp_t[:] customer_addr_idx,
                 customer_arrays, **kwargs):
    new_sol = solution.copy()

    cdef list candidates = []
    cdef list route
    for route in new_sol.routes[:-1]:
        candidates.extend(route)

    if not candidates:
        return new_sol

    cdef double[:] tw_start = customer_arrays['tw_start']
    cdef double[:] tw_end   = customer_arrays['tw_end']
    cdef double[:] demand   = customer_arrays['demand']

    cdef int seed = candidates[c_rand_int(len(candidates))]
    cdef set removed = {seed}

    cdef double w_dist = 9.0
    cdef double w_time = 3.0
    cdef double w_dem  = 2.0

    cdef double max_dist = np.max(distance_matrix_array) if np.max(distance_matrix_array) > 0 else 1.0
    cdef double max_time = np.max(tw_end) if np.max(tw_end) > 0 else 1.0
    cdef double max_dem  = np.max(demand) if np.max(demand) > 0 else 1.0

    cdef int ref_cust, ref_idx, cust, c_idx
    cdef double ref_tw, ref_d
    cdef double dist_val, time_val, dem_val, relatedness
    cdef list sample_pool, scored
    cdef int pick_idx

    while len(removed) < <unsigned int>num_to_remove and len(removed) < <unsigned int>len(candidates):
        ref_cust = py_random.choice(list(removed))
        ref_idx  = customer_addr_idx[ref_cust - 1]
        ref_tw   = tw_start[ref_cust - 1]
        ref_d    = demand[ref_cust - 1]

        sample_pool = [c for c in candidates if c not in removed]
        if not sample_pool:
            break
        if len(sample_pool) > 50:
            sample_pool = py_random.sample(sample_pool, 50)

        scored = []
        for cust in sample_pool:
            c_idx = customer_addr_idx[cust - 1]
            dist_val = distance_matrix_array[ref_idx, c_idx] / max_dist
            time_val = fabs(ref_tw - tw_start[cust - 1]) / max_time
            dem_val  = fabs(ref_d  - demand[cust - 1])    / max_dem
            relatedness = w_dist * dist_val + w_time * time_val + w_dem * dem_val
            scored.append((relatedness, cust))

        scored.sort(key=lambda x: x[0])

        if scored:
            pick_idx = c_rand_int(min(3, len(scored)))
            removed.add(scored[pick_idx][1])
        else:
            break

    cdef int i
    for i in range(len(new_sol.routes) - 1):
        new_sol.routes[i] = [c for c in new_sol.routes[i] if c not in removed]

    new_sol.routes[len(new_sol.routes)-1].extend(list(removed))
    return new_sol


def least_used_vehicle_removal(solution, int num_to_remove, **kwargs):
    new_sol = solution.copy()

    cdef list routes_info = []
    cdef int i, r_len
    cdef list route

    for i in range(len(new_sol.routes) - 1):
        route = new_sol.routes[i]
        r_len = len(route)
        if r_len > 0:
            routes_info.append((i, r_len))

    py_random.shuffle(routes_info)
    routes_info.sort(key=lambda x: x[1])

    cdef set targets = set()
    cdef int current_removed = 0
    cdef int r_idx

    for r_idx, r_len in routes_info:
        if current_removed + r_len <= num_to_remove:
            targets.update(new_sol.routes[r_idx])
            current_removed += r_len
        elif current_removed == 0:
            subset = py_random.sample(new_sol.routes[r_idx], num_to_remove)
            targets.update(subset)
            break
        else:
            break

    for i in range(len(new_sol.routes) - 1):
        new_sol.routes[i] = [c for c in new_sol.routes[i] if c not in targets]

    new_sol.routes[len(new_sol.routes)-1].extend(list(targets))
    return new_sol


# =============================================================
#                SOFTMAX HELPERS
# =============================================================

cdef inline void _softmax_select(double[:] costs_view, int n,
                                 double temperature,
                                 double* out_probs) noexcept nogil:
    cdef double min_c = costs_view[0]
    cdef int i
    for i in range(1, n):
        if costs_view[i] < min_c:
            min_c = costs_view[i]

    cdef double total = 0.0
    cdef double inv_temp = 1.0 / (temperature + 1e-9)
    for i in range(n):
        out_probs[i] = exp(-(costs_view[i] - min_c) * inv_temp)
        total += out_probs[i]

    cdef double inv_total = 1.0 / total
    for i in range(n):
        out_probs[i] *= inv_total


cdef int _weighted_choice(double* probs, int n) noexcept nogil:
    cdef double r = c_rand_double()
    cdef double cumsum = 0.0
    cdef int i
    for i in range(n):
        cumsum += probs[i]
        if r < cumsum:
            return i
    return n - 1


# =============================================================
#                      REPAIR  OPERATORS
# =============================================================

def greedy_insertion(solution,
                     double[:, :] distance_matrix_array,
                     np.intp_t[:] customer_addr_idx,
                     customer_arrays,
                     vehicles_df,
                     neighbor_sets,
                     int depot_idx=0,
                     double temperature=1.0,
                     **kwargs):
    """
    Greedy insertion with Blended Softmax selection.
    Feasibility is now checked via inlined C — no Python calls in the hot loop.
    """
    new_sol = solution.copy()
    cdef list unassigned = list(new_sol.routes[len(new_sol.routes)-1])
    new_sol.routes[len(new_sol.routes)-1] = []

    py_random.shuffle(unassigned)

    cdef double capacity
    if isinstance(vehicles_df, dict):
        capacity = vehicles_df['capacity']
    else:
        capacity = vehicles_df.loc['Standard', 'capacity']

    # Pre-extract ALL typed memoryviews once
    cdef double[:] demands      = customer_arrays['demand']
    cdef double[:] tw_start     = customer_arrays['tw_start']
    cdef double[:] tw_end       = customer_arrays['tw_end']
    cdef double[:] service_time = customer_arrays['service_time']

    cdef int cust, cust_addr, r_idx, i, n_routes, route_len
    cdef double cust_demand, route_demand, d
    cdef int prev_addr, next_addr
    cdef list route
    cdef list feasible_points
    cdef int n_fp, idx
    cdef int selected_r, selected_p
    cdef double[:] cost_view

    cdef int max_fp = 4096
    cdef double[4096] probs_buf

    for cust in unassigned:
        cust_addr   = customer_addr_idx[cust - 1]
        cust_demand = demands[cust - 1]
        feasible_points = []

        n_routes = len(new_sol.routes) - 1
        for r_idx in range(n_routes):
            route = new_sol.routes[r_idx]
            route_len = len(route)

            # ── Fast capacity pre-check (typed loop) ──
            route_demand = 0.0
            for i in range(route_len):
                route_demand += demands[route[i] - 1]
            if route_demand + cust_demand > capacity:
                continue

            # ── Evaluate every insertion position ──
            for i in range(route_len + 1):
                prev_addr = depot_idx if i == 0 else customer_addr_idx[route[i - 1] - 1]
                next_addr = depot_idx if i == route_len else customer_addr_idx[route[i] - 1]

                d = delta_cost(distance_matrix_array, prev_addr, cust_addr, next_addr)

                # ── INLINED C feasibility — no list construction ──
                if c_check_tw_with_insertion(
                        route, route_len, i, cust,
                        distance_matrix_array, customer_addr_idx,
                        tw_start, tw_end, service_time, depot_idx):
                    feasible_points.append((d, r_idx, i))

        # ── Softmax selection ──
        n_fp = len(feasible_points)
        if n_fp > 0:
            cost_arr = np.empty(n_fp, dtype=np.float64)
            cost_view = cost_arr
            for i in range(n_fp):
                cost_view[i] = feasible_points[i][0]

            if n_fp <= max_fp:
                _softmax_select(cost_view, n_fp, temperature, probs_buf)
                idx = _weighted_choice(probs_buf, n_fp)
            else:
                norm = (cost_arr - cost_arr.min()) / (temperature + 1e-9)
                p = np.exp(-norm)
                p /= p.sum()
                idx = np.random.choice(n_fp, p=p)

            _, selected_r, selected_p = feasible_points[idx]
            new_sol.routes[selected_r].insert(selected_p, cust)
        else:
            new_sol.routes[len(new_sol.routes)-1].append(cust)

    return new_sol


def regret_insertion(solution,
                     double[:, :] distance_matrix_array,
                     np.intp_t[:] customer_addr_idx,
                     customer_arrays,
                     vehicles_df,
                     neighbor_sets,
                     int depot_idx=0,
                     double temperature=1.0,
                     **kwargs):
    """
    2-Regret Insertion with Blended Softmax for position selection.
    Feasibility inlined as C — no Python calls in the hot loop.
    """
    new_sol = solution.copy()
    cdef list unassigned = list(new_sol.routes[len(new_sol.routes)-1])
    new_sol.routes[len(new_sol.routes)-1] = []

    cdef double capacity
    if isinstance(vehicles_df, dict):
        capacity = vehicles_df['capacity']
    else:
        capacity = vehicles_df.loc['Standard', 'capacity']

    cdef double[:] demands      = customer_arrays['demand']
    cdef double[:] tw_start     = customer_arrays['tw_start']
    cdef double[:] tw_end       = customer_arrays['tw_end']
    cdef double[:] service_time = customer_arrays['service_time']

    cdef int cust, cust_addr, r_idx, i, route_len, n_routes
    cdef double cust_demand, route_demand, d
    cdef int prev_addr, next_addr
    cdef double regret_val
    cdef list route, feasible_options
    cdef int n_opts, choice_idx
    cdef int final_r, final_p
    cdef double[:] cost_view

    cdef int max_fp = 4096
    cdef double[4096] probs_buf

    while unassigned:
        potential_insertions = []

        for cust in unassigned:
            cust_addr   = customer_addr_idx[cust - 1]
            cust_demand = demands[cust - 1]
            feasible_options = []

            n_routes = len(new_sol.routes) - 1
            for r_idx in range(n_routes):
                route = new_sol.routes[r_idx]
                route_len = len(route)

                route_demand = 0.0
                for i in range(route_len):
                    route_demand += demands[route[i] - 1]
                if route_demand + cust_demand > capacity:
                    continue

                for i in range(route_len + 1):
                    prev_addr = depot_idx if i == 0 else customer_addr_idx[route[i - 1] - 1]
                    next_addr = depot_idx if i == route_len else customer_addr_idx[route[i] - 1]

                    d = delta_cost(distance_matrix_array, prev_addr, cust_addr, next_addr)

                    # ── INLINED C feasibility — no list construction ──
                    if c_check_tw_with_insertion(
                            route, route_len, i, cust,
                            distance_matrix_array, customer_addr_idx,
                            tw_start, tw_end, service_time, depot_idx):
                        feasible_options.append((d, r_idx, i))

            if not feasible_options:
                continue

            feasible_options.sort(key=lambda x: x[0])

            if len(feasible_options) >= 2:
                regret_val = feasible_options[1][0] - feasible_options[0][0]
            else:
                regret_val = 1e6

            potential_insertions.append((regret_val, cust, feasible_options))

        if not potential_insertions:
            new_sol.routes[len(new_sol.routes)-1].extend(unassigned)
            break

        potential_insertions.sort(key=lambda x: x[0], reverse=True)
        best = potential_insertions[0]
        target_cust      = best[1]
        available_options = best[2]

        n_opts = len(available_options)
        cost_arr = np.empty(n_opts, dtype=np.float64)
        cost_view = cost_arr
        for i in range(n_opts):
            cost_view[i] = available_options[i][0]

        if n_opts <= max_fp:
            _softmax_select(cost_view, n_opts, temperature, probs_buf)
            choice_idx = _weighted_choice(probs_buf, n_opts)
        else:
            norm = (cost_arr - cost_arr.min()) / (temperature + 1e-9)
            p = np.exp(-norm)
            p /= p.sum()
            choice_idx = np.random.choice(n_opts, p=p)

        _, final_r, final_p = available_options[choice_idx]
        new_sol.routes[final_r].insert(final_p, target_cust)
        unassigned.remove(target_cust)

    return new_sol
