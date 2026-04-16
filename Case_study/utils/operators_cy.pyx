# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# NOTE: To debug segfaults, temporarily set boundscheck=True and wraparound=True

"""
Cythonized ALNS Destroy & Repair Operators + Feasibility for VRPTW Case Study.

Combines operators.py and feasibility.py into a single .pyx for maximum
speed: all hot-loop feasibility checks are inlined as cdef C functions.

Key differences from Benchmark version:
- 4-dim capacity (PPL, freezer, volume, weight) with 1/3 PPL tolerance
- Shift-aware departure (earliest_departure + LOADING_TIME)
- Lunch break placement (30 min within first 6h of shift)
- Vehicle-store compatibility (biltype check)
- Two-phase repair (trip-1 first, then trip-2)
- Route metadata (shift/trip info per route slot)
"""

import random as py_random
import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.math cimport exp, fabs
from libc.time cimport time as c_time

# Seed the C RNG once at import time
srand(<unsigned int>c_time(NULL))

# Type aliases
ctypedef np.float64_t DTYPE_f
ctypedef np.intp_t    DTYPE_i

# ── Constants (must match feasibility.py) ──
DEF LOADING_TIME_C = 1.0
DEF DELOADING_TIME_C = 0.333333333   # 20 min = 1/3 hour
DEF LUNCH_DURATION_C = 0.5
DEF LUNCH_DEADLINE_HOURS_C = 6.0

# Python-visible constants (so alns.py can import them)
LOADING_TIME = LOADING_TIME_C
DELOADING_TIME = DELOADING_TIME_C
LUNCH_DURATION = LUNCH_DURATION_C


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
#      INLINED FEASIBILITY CHECKS  (cdef — C speed)
# =============================================================

cdef double c_compute_return_time(list route, int route_len,
                                  double[:, :] dm,
                                  np.intp_t[:] addr_idx,
                                  double[:] tw_start,
                                  double[:] service_time,
                                  int depot_idx,
                                  double earliest_dep,
                                  int lunch_pos,
                                  double lunch_dur):
    """Compute when vehicle returns to depot after completing the route."""
    if route_len == 0:
        return earliest_dep

    cdef double current_time = earliest_dep + LOADING_TIME_C
    cdef int last_addr = depot_idx
    cdef int i, cust, curr_addr
    cdef double tw_s, svc

    for i in range(route_len):
        cust = <int>route[i]
        curr_addr = addr_idx[cust - 1]
        current_time += dm[last_addr, curr_addr]
        tw_s = tw_start[cust - 1]
        if current_time < tw_s:
            current_time = tw_s
        svc = service_time[cust - 1]
        current_time += svc
        last_addr = curr_addr
        # Lunch break after this customer
        if lunch_pos > 0 and lunch_pos == i + 1:
            current_time += lunch_dur

    # Return to depot
    current_time += dm[last_addr, depot_idx]
    return current_time


cdef bint c_check_tw_with_insertion(list route, int route_len,
                                     int insert_pos, int new_cust,
                                     double[:, :] dm,
                                     np.intp_t[:] addr_idx,
                                     double[:] tw_start,
                                     double[:] tw_end,
                                     double[:] service_time,
                                     int depot_idx,
                                     double earliest_dep):
    """
    Check time-window feasibility for route WITH a customer inserted at insert_pos,
    WITHOUT building a new Python list.
    Uses shift-aware departure: earliest_dep + LOADING_TIME.
    """
    cdef double current_time = earliest_dep + LOADING_TIME_C
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


cdef bint c_check_tw_route(list route, int route_len,
                            double[:, :] dm,
                            np.intp_t[:] addr_idx,
                            double[:] tw_start,
                            double[:] tw_end,
                            double[:] service_time,
                            int depot_idx,
                            double earliest_dep):
    """Check time-window feasibility for an existing route (no insertion)."""
    if route_len == 0:
        return True

    cdef double current_time = earliest_dep + LOADING_TIME_C
    cdef int current_loc = depot_idx
    cdef int i, cust, cust_addr
    cdef double arrival

    for i in range(route_len):
        cust = <int>route[i]
        cust_addr = addr_idx[cust - 1]
        arrival = current_time + dm[current_loc, cust_addr]

        if arrival > tw_end[cust - 1]:
            return False

        if arrival < tw_start[cust - 1]:
            arrival = tw_start[cust - 1]

        current_time = arrival + service_time[cust - 1]
        current_loc = cust_addr

    return True


cdef bint c_check_shift(double return_time, double deload, double lunch_dur,
                         double shift_end) noexcept nogil:
    """Check that route + deload + lunch fits within shift window."""
    return return_time + deload + lunch_dur <= shift_end + 0.01


cdef int c_find_lunch_position(list route, int route_len,
                                double[:, :] dm,
                                np.intp_t[:] addr_idx,
                                double[:] tw_start,
                                double[:] tw_end,
                                double[:] service_time,
                                int depot_idx,
                                double lunch_dur,
                                double earliest_dep,
                                double shift_start):
    """
    Find earliest feasible lunch position (1..n) in the route.
    Returns position (1-based) or 0 if no valid placement.
    """
    if route_len == 0:
        return 0

    cdef int break_pos
    for break_pos in range(1, route_len + 1):
        if c_simulate_with_break(route, route_len, break_pos, dm, addr_idx,
                                  tw_start, tw_end, service_time, depot_idx,
                                  lunch_dur, earliest_dep, shift_start):
            return break_pos
    return 0


cdef bint c_simulate_with_break(list route, int route_len, int break_pos,
                                 double[:, :] dm,
                                 np.intp_t[:] addr_idx,
                                 double[:] tw_start,
                                 double[:] tw_end,
                                 double[:] service_time,
                                 int depot_idx,
                                 double lunch_dur,
                                 double earliest_dep,
                                 double shift_start):
    """
    Simulate route timing with lunch break at break_pos (after k-th customer).
    Returns True if all time windows are respected and lunch is within deadline.
    """
    cdef double current_time = earliest_dep + LOADING_TIME_C
    cdef int last_addr = depot_idx
    cdef int i, cust, curr_addr
    cdef double tw_s, tw_e, svc

    for i in range(route_len):
        cust = <int>route[i]
        curr_addr = addr_idx[cust - 1]
        current_time += dm[last_addr, curr_addr]

        tw_e = tw_end[cust - 1]
        if current_time > tw_e:
            return False

        tw_s = tw_start[cust - 1]
        if current_time < tw_s:
            current_time = tw_s
        svc = service_time[cust - 1]
        current_time += svc
        last_addr = curr_addr

        if break_pos == i + 1:
            # Lunch deadline check
            if current_time > shift_start + LUNCH_DEADLINE_HOURS_C:
                return False
            current_time += lunch_dur

    return True


cdef bint c_check_capacity(list route, int route_len, int new_cust,
                            double[:] demand, double[:] frys,
                            double[:] vol, double[:] weight,
                            double cap_ppl, double cap_frys,
                            double cap_vol, double cap_wt):
    """4-dimensional capacity check with 1/3 PPL tolerance."""
    cdef double sum_ppl = 0.0, sum_frys = 0.0, sum_vol = 0.0, sum_wt = 0.0
    cdef int i, c
    cdef double ppl_tol = 0.333333 - 1e-6
    cdef double tol = 1e-6

    for i in range(route_len):
        c = <int>route[i] - 1
        sum_ppl += demand[c]
        sum_frys += frys[c]
        sum_vol += vol[c]
        sum_wt += weight[c]

    if new_cust > 0:
        c = new_cust - 1
        sum_ppl += demand[c]
        sum_frys += frys[c]
        sum_vol += vol[c]
        sum_wt += weight[c]

    if sum_ppl > cap_ppl + ppl_tol:
        return False
    if sum_frys > cap_frys + ppl_tol:
        return False
    if sum_vol > cap_vol + tol:
        return False
    if sum_wt > cap_wt + tol:
        return False
    return True


cdef bint c_check_biltype(list route, int route_len, int new_cust,
                           np.int32_t[:] biltype_arr,
                           double vehicle_ppl, set compatible_ppls):
    """Vehicle-store compatibility: biltype=2 customers need small/medium vehicles."""
    cdef int i, c, bt

    for i in range(route_len):
        c = <int>route[i] - 1
        bt = biltype_arr[c]
        if bt == 2 and vehicle_ppl not in compatible_ppls:
            return False
        if bt > 2:
            return False

    if new_cust > 0:
        c = new_cust - 1
        bt = biltype_arr[c]
        if bt == 2 and vehicle_ppl not in compatible_ppls:
            return False
        if bt > 2:
            return False

    return True


# =============================================================
#      SOFTMAX HELPERS
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
#      HELPER: get_earliest_departure (Python-visible)
# =============================================================

def get_earliest_departure(solution, int r_idx,
                           double[:, :] time_matrix_array,
                           np.intp_t[:] customer_addr_idx,
                           customer_arrays, int depot_idx):
    """
    Get earliest time loading can start for this trip slot.
    Trip 1: shift_start.  Trip N>1: end of trip N-1 + deload.
    """
    cdef int prev_idx
    cdef double[:] tw_start_v
    cdef double[:] service_time_v
    cdef int lunch_pos_val = 0
    cdef double ret, prev_departure

    meta = solution.route_meta[r_idx]
    if meta is None:
        return 0.0

    if meta['trip'] == 1:
        return meta['shift_start']

    # Trip N>1: find previous trip on same vehicle (preceding slot)
    prev_idx = r_idx - 1
    prev_meta = solution.route_meta[prev_idx]
    if (prev_meta is not None and
            prev_meta['vehicle_idx'] == meta['vehicle_idx'] and
            prev_meta['trip'] == meta['trip'] - 1):
        prev_route = solution.routes[prev_idx]
        if not prev_route:
            # Previous trip empty — recurse to find latest non-empty trip
            return get_earliest_departure(solution, prev_idx, time_matrix_array,
                                          customer_addr_idx, customer_arrays, depot_idx)

        # Compute when previous trip's loading started
        prev_departure = get_earliest_departure(solution, prev_idx, time_matrix_array,
                                                customer_addr_idx, customer_arrays, depot_idx)

        tw_start_v = customer_arrays['tw_start']
        service_time_v = customer_arrays['service_time']
        lb = solution.lunch_breaks[prev_idx]
        if lb is not None:
            lunch_pos_val = <int>lb

        ret = c_compute_return_time(
            prev_route, len(prev_route), time_matrix_array,
            customer_addr_idx, tw_start_v, service_time_v,
            depot_idx, prev_departure, lunch_pos_val, LUNCH_DURATION_C
        )
        return ret + DELOADING_TIME_C

    return meta['shift_start']


cdef double c_get_earliest_departure(solution, int r_idx,
                                      double[:, :] dm,
                                      np.intp_t[:] addr_idx,
                                      double[:] tw_start,
                                      double[:] service_time,
                                      int depot_idx):
    """C-level version of get_earliest_departure for internal use."""
    cdef int prev_idx
    cdef int lunch_pos_val = 0
    cdef double ret, prev_departure

    meta = solution.route_meta[r_idx]
    if meta is None:
        return 0.0

    if meta['trip'] == 1:
        return <double>meta['shift_start']

    prev_idx = r_idx - 1
    prev_meta = solution.route_meta[prev_idx]
    if (prev_meta is not None and
            prev_meta['vehicle_idx'] == meta['vehicle_idx'] and
            prev_meta['trip'] == <int>meta['trip'] - 1):
        prev_route = solution.routes[prev_idx]
        if not prev_route:
            return c_get_earliest_departure(solution, prev_idx, dm, addr_idx,
                                            tw_start, service_time, depot_idx)

        prev_departure = c_get_earliest_departure(solution, prev_idx, dm, addr_idx,
                                                  tw_start, service_time, depot_idx)

        lb = solution.lunch_breaks[prev_idx]
        if lb is not None:
            lunch_pos_val = <int>lb

        ret = c_compute_return_time(
            prev_route, len(prev_route), dm, addr_idx,
            tw_start, service_time, depot_idx,
            prev_departure, lunch_pos_val, LUNCH_DURATION_C
        )
        return ret + DELOADING_TIME_C

    return <double>meta['shift_start']


# =============================================================
#      Python-visible feasibility wrappers (for validate_and_trim)
# =============================================================

def find_lunch_break_position(route_indices, time_matrix_array, customer_addr_idx,
                               customer_arrays, depot_idx, lunch_duration=0.5,
                               earliest_departure=None, shift_start=None):
    """Python-visible wrapper for c_find_lunch_position."""
    if not route_indices:
        return None

    cdef double[:, :] dm = time_matrix_array
    cdef np.intp_t[:] addr_idx = customer_addr_idx
    cdef double[:] tw_s = customer_arrays['tw_start']
    cdef double[:] tw_e = customer_arrays['tw_end']
    cdef double[:] svc = customer_arrays['service_time']
    cdef double ed = earliest_departure if earliest_departure is not None else 0.0
    cdef double ss = shift_start if shift_start is not None else 6.0
    cdef int dep = depot_idx

    cdef int result = c_find_lunch_position(
        route_indices, len(route_indices), dm, addr_idx,
        tw_s, tw_e, svc, dep, lunch_duration, ed, ss
    )
    return result if result > 0 else None


def check_lunch_break_feasibility(route_indices, time_matrix_array, customer_addr_idx,
                                   customer_arrays, depot_idx, lunch_duration=0.5,
                                   earliest_departure=None, shift_start=None, debug=False):
    """Python-visible wrapper."""
    if not route_indices:
        return True
    result = find_lunch_break_position(route_indices, time_matrix_array, customer_addr_idx,
                                        customer_arrays, depot_idx, lunch_duration,
                                        earliest_departure, shift_start)
    return result is not None


# =============================================================
#      VALIDATE AND TRIM (post-repair safety net)
# =============================================================

def validate_and_trim_routes(solution, time_matrix_array, customer_addr_idx,
                              customer_arrays, depot_idx, verbose=False):
    """
    Safety-net: check every route for max trip duration feasibility. If overflow,
    remove customers from end until it fits. Moved customers go to dummy.
    """
    cdef double[:, :] dm = time_matrix_array
    cdef np.intp_t[:] addr_idx = customer_addr_idx
    cdef double[:] tw_start_v = customer_arrays['tw_start']
    cdef double[:] tw_end_v = customer_arrays['tw_end']
    cdef double[:] service_time_v = customer_arrays['service_time']
    cdef int dep = depot_idx

    cdef int moved = 0
    cdef int r_idx, route_len
    cdef double max_trip_h, earliest_dep, return_time, end_time
    cdef int is_trip1, lunch_pos_val
    cdef double lunch_dur

    for r_idx in range(len(solution.routes) - 1):
        route = solution.routes[r_idx]
        if not route:
            continue
        meta = solution.route_meta[r_idx]
        if meta is None:
            continue

        max_trip_h = <double>meta.get('max_trip_hours', 8.0)
        is_trip1 = 1 if meta['trip'] == 1 else 0

        earliest_dep = c_get_earliest_departure(
            solution, r_idx, dm, addr_idx, tw_start_v, service_time_v, dep
        )

        lunch_pos_val = 0
        if is_trip1:
            lb = solution.lunch_breaks[r_idx]
            if lb is not None:
                lunch_pos_val = <int>lb
        lunch_dur = LUNCH_DURATION_C if is_trip1 else 0.0

        route_len = len(route)
        return_time = c_compute_return_time(
            route, route_len, dm, addr_idx, tw_start_v, service_time_v,
            dep, earliest_dep, lunch_pos_val, lunch_dur
        )
        end_time = return_time + DELOADING_TIME_C

        while route and end_time > earliest_dep + max_trip_h + 0.01:
            removed_cust = route.pop()
            solution.routes[len(solution.routes) - 1].append(removed_cust)
            moved += 1
            if not route:
                solution.lunch_breaks[r_idx] = None
                break

            route_len = len(route)
            if is_trip1:
                lunch_pos_val = c_find_lunch_position(
                    route, route_len, dm, addr_idx,
                    tw_start_v, tw_end_v, service_time_v,
                    dep, LUNCH_DURATION_C, earliest_dep, <double>meta['shift_start']
                )
                solution.lunch_breaks[r_idx] = lunch_pos_val if lunch_pos_val > 0 else None
            else:
                lunch_pos_val = 0

            return_time = c_compute_return_time(
                route, route_len, dm, addr_idx, tw_start_v, service_time_v,
                dep, earliest_dep, lunch_pos_val, lunch_dur if is_trip1 else 0.0
            )
            end_time = return_time + DELOADING_TIME_C

    return moved


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

    new_sol.routes[len(new_sol.routes) - 1].extend(list(to_remove))
    return new_sol


def worst_removal(solution, int num_to_remove,
                  double[:, :] time_matrix_array,
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

            removed_cost = time_matrix_array[prev_addr, cust_addr] + \
                           time_matrix_array[cust_addr, next_addr]
            added_cost = time_matrix_array[prev_addr, next_addr]
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

    new_sol.routes[len(new_sol.routes) - 1].extend(list(targets))
    return new_sol


def cluster_removal(solution, int num_to_remove,
                    double[:, :] time_matrix_array,
                    np.intp_t[:] customer_addr_idx, **kwargs):
    new_sol = solution.copy()

    cdef list candidates = []
    cdef list route
    for route in new_sol.routes[:-1]:
        candidates.extend(route)

    if not candidates:
        return new_sol

    cdef int center_cust = candidates[c_rand_int(len(candidates))]
    cdef int center_idx = customer_addr_idx[center_cust - 1]

    cdef list distances = []
    cdef int cust, c_idx
    cdef double dist

    for cust in candidates:
        if cust == center_cust:
            continue
        c_idx = customer_addr_idx[cust - 1]
        dist = time_matrix_array[center_idx, c_idx]
        distances.append((dist, cust))

    distances.sort(key=lambda x: x[0])

    cdef set targets = {center_cust}
    cdef int j
    for j in range(min(num_to_remove - 1, len(distances))):
        targets.add(distances[j][1])

    cdef int i
    for i in range(len(new_sol.routes) - 1):
        new_sol.routes[i] = [c for c in new_sol.routes[i] if c not in targets]

    new_sol.routes[len(new_sol.routes) - 1].extend(list(targets))
    return new_sol


def shaw_removal(solution, int num_to_remove,
                 double[:, :] time_matrix_array,
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
    cdef double[:] tw_end = customer_arrays['tw_end']
    cdef double[:] demand = customer_arrays['demand']

    cdef int seed = candidates[c_rand_int(len(candidates))]
    cdef set removed = {seed}

    cdef double w_dist = 9.0, w_time = 3.0, w_dem = 2.0

    cdef double max_dist = np.max(time_matrix_array) if np.max(time_matrix_array) > 0 else 1.0
    cdef double max_time = np.max(tw_end) if np.max(tw_end) > 0 else 1.0
    cdef double max_dem = np.max(demand) if np.max(demand) > 0 else 1.0

    cdef int ref_cust, ref_idx, cust, c_idx
    cdef double ref_tw, ref_d
    cdef double dist_val, time_val, dem_val, relatedness
    cdef list sample_pool, scored
    cdef int pick_idx

    while len(removed) < <unsigned int>num_to_remove and len(removed) < <unsigned int>len(candidates):
        ref_cust = py_random.choice(list(removed))
        ref_idx = customer_addr_idx[ref_cust - 1]
        ref_tw = tw_start[ref_cust - 1]
        ref_d = demand[ref_cust - 1]

        sample_pool = [c for c in candidates if c not in removed]
        if not sample_pool:
            break
        if len(sample_pool) > 50:
            sample_pool = py_random.sample(sample_pool, 50)

        scored = []
        for cust in sample_pool:
            c_idx = customer_addr_idx[cust - 1]
            dist_val = time_matrix_array[ref_idx, c_idx] / max_dist
            time_val = fabs(ref_tw - tw_start[cust - 1]) / max_time
            dem_val = fabs(ref_d - demand[cust - 1]) / max_dem
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

    new_sol.routes[len(new_sol.routes) - 1].extend(list(removed))
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

    new_sol.routes[len(new_sol.routes) - 1].extend(list(targets))
    return new_sol


# =============================================================
#      TRIP-2 VALIDATION (internal helper)
# =============================================================

cdef void _validate_later_trips(solution,
                                 double[:, :] dm,
                                 np.intp_t[:] addr_idx,
                                 double[:] tw_start,
                                 double[:] tw_end,
                                 double[:] service_time,
                                 int depot_idx):
    """After repair, check all trip N>1 routes; move infeasible ones to dummy."""
    cdef int r_idx, route_len
    cdef double earliest_dep
    cdef list route

    for r_idx in range(len(solution.routes) - 1):
        meta = solution.route_meta[r_idx]
        if meta is None or meta['trip'] <= 1:
            continue
        route = solution.routes[r_idx]
        if not route:
            continue

        earliest_dep = c_get_earliest_departure(
            solution, r_idx, dm, addr_idx, tw_start, service_time, depot_idx
        )
        route_len = len(route)

        # Check time windows with updated departure
        feasible = c_check_tw_route(
            route, route_len, dm, addr_idx, tw_start, tw_end,
            service_time, depot_idx, earliest_dep
        )

        if not feasible:
            solution.routes[len(solution.routes) - 1].extend(route)
            solution.routes[r_idx] = []
            solution.lunch_breaks[r_idx] = None


# =============================================================
#                      REPAIR  OPERATORS
# =============================================================

def greedy_insertion(solution,
                     double[:, :] time_matrix_array,
                     np.intp_t[:] customer_addr_idx,
                     customer_arrays,
                     vehicles_dict,
                     neighbor_sets,
                     int depot_idx=0,
                     double temperature=1.0,
                     **kwargs):
    """Greedy insertion with two-phase approach: trip-1 first, then trip-2."""
    new_sol = solution.copy()
    cdef list unassigned = list(new_sol.routes[len(new_sol.routes) - 1])
    new_sol.routes[len(new_sol.routes) - 1] = []
    compatible_ppls_set = kwargs.get('compatible_ppls_set', set())

    # Pre-extract typed memoryviews
    cdef double[:] tw_start = customer_arrays['tw_start']
    cdef double[:] tw_end = customer_arrays['tw_end']
    cdef double[:] service_time = customer_arrays['service_time']
    cdef double[:] demand_arr = customer_arrays['demand']
    cdef double[:] frys_arr = customer_arrays['frys']
    cdef double[:] vol_arr = customer_arrays['volume_m3']
    cdef double[:] wt_arr = customer_arrays['weight_kg']
    cdef np.int32_t[:] biltype_arr = customer_arrays['biltype']

    cdef int cust, cust_addr, r_idx, i, n_routes, route_len
    cdef double d, earliest_dep, shift_start_val
    cdef double cap_ppl, cap_frys, cap_vol, cap_wt, veh_ppl
    cdef int prev_addr, next_addr
    cdef list route, feasible_points, still_unassigned
    cdef int n_fp, idx
    cdef int selected_r, selected_p
    cdef double[:] cost_view
    cdef int lunch_pos_val
    cdef int phase_trip, is_trip1

    cdef int max_fp = 4096
    cdef double[4096] probs_buf
    cdef int max_trips = 4  # MAX_TRIPS

    for phase_trip in range(1, max_trips + 1):
        if phase_trip > 1:
            _validate_later_trips(new_sol, time_matrix_array, customer_addr_idx,
                                   tw_start, tw_end, service_time, depot_idx)
            unassigned.extend(new_sol.routes[len(new_sol.routes) - 1])
            new_sol.routes[len(new_sol.routes) - 1] = []

        still_unassigned = []
        for cust in unassigned:
            cust_addr = customer_addr_idx[cust - 1]
            feasible_points = []

            n_routes = len(new_sol.routes) - 1
            for r_idx in range(n_routes):
                route = new_sol.routes[r_idx]
                route_len = len(route)
                meta = new_sol.route_meta[r_idx]

                # Phase filter
                if meta is None or meta['trip'] != phase_trip:
                    continue
                if phase_trip > 1 and not new_sol.routes[r_idx - 1]:
                    continue

                vehicle_name = new_sol.vehicles[r_idx]
                cap = vehicles_dict[vehicle_name]
                cap_ppl = cap['PPL total']
                cap_frys = cap['PPL Frys']
                cap_vol = cap['m3']
                cap_wt = cap['Vekt (KG)']
                veh_ppl = cap_ppl

                # Capacity check
                if not c_check_capacity(route, route_len, cust, demand_arr, frys_arr,
                                         vol_arr, wt_arr, cap_ppl, cap_frys, cap_vol, cap_wt):
                    continue

                # Biltype check
                if not c_check_biltype(route, route_len, cust, biltype_arr,
                                        veh_ppl, compatible_ppls_set):
                    continue

                earliest_dep = c_get_earliest_departure(
                    new_sol, r_idx, time_matrix_array, customer_addr_idx,
                    tw_start, service_time, depot_idx
                )
                shift_start_val = <double>meta['shift_start']
                max_trip_h = <double>meta['max_trip_hours']
                is_trip1 = 1 if phase_trip == 1 else 0
                lunch_buf = LUNCH_DURATION_C if is_trip1 else 0.0

                for i in range(route_len + 1):
                    prev_addr = depot_idx if i == 0 else customer_addr_idx[route[i - 1] - 1]
                    next_addr = depot_idx if i == route_len else customer_addr_idx[route[i] - 1]

                    d = delta_cost(time_matrix_array, prev_addr, cust_addr, next_addr)

                    # Inlined TW check with insertion
                    if not c_check_tw_with_insertion(
                            route, route_len, i, cust,
                            time_matrix_array, customer_addr_idx,
                            tw_start, tw_end, service_time,
                            depot_idx, earliest_dep):
                        continue

                    # Trip duration check
                    candidate_route = route[:i] + [cust] + route[i:]
                    cand_len = route_len + 1
                    return_time = c_compute_return_time(
                        candidate_route, cand_len, time_matrix_array,
                        customer_addr_idx, tw_start, service_time,
                        depot_idx, earliest_dep, 0, 0.0
                    )
                    if not c_check_shift(return_time, DELOADING_TIME_C,
                                          lunch_buf, earliest_dep + max_trip_h):
                        continue

                    # Lunch feasibility for trip-1
                    if is_trip1:
                        lunch_pos_val = c_find_lunch_position(
                            candidate_route, cand_len, time_matrix_array,
                            customer_addr_idx, tw_start, tw_end, service_time,
                            depot_idx, LUNCH_DURATION_C, earliest_dep, shift_start_val
                        )
                        if lunch_pos_val == 0:
                            continue

                    feasible_points.append((d, r_idx, i))

            # Softmax selection
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

                # Update lunch position for trip-1
                sel_meta = new_sol.route_meta[selected_r]
                if phase_trip == 1:
                    sel_dep = c_get_earliest_departure(
                        new_sol, selected_r, time_matrix_array, customer_addr_idx,
                        tw_start, service_time, depot_idx
                    )
                    sel_route = new_sol.routes[selected_r]
                    lunch_pos_val = c_find_lunch_position(
                        sel_route, len(sel_route), time_matrix_array,
                        customer_addr_idx, tw_start, tw_end, service_time,
                        depot_idx, LUNCH_DURATION_C, sel_dep, <double>sel_meta['shift_start']
                    )
                    new_sol.lunch_breaks[selected_r] = lunch_pos_val if lunch_pos_val > 0 else None
                else:
                    new_sol.lunch_breaks[selected_r] = None
            else:
                still_unassigned.append(cust)

        unassigned = still_unassigned

    new_sol.routes[len(new_sol.routes) - 1] = unassigned
    return new_sol


def regret_insertion(solution,
                     double[:, :] time_matrix_array,
                     np.intp_t[:] customer_addr_idx,
                     customer_arrays,
                     vehicles_dict,
                     neighbor_sets,
                     int depot_idx=0,
                     double temperature=1.0,
                     **kwargs):
    """2-Regret insertion with two-phase approach: trip-1 first, then trip-2."""
    new_sol = solution.copy()
    cdef list unassigned = list(new_sol.routes[len(new_sol.routes) - 1])
    new_sol.routes[len(new_sol.routes) - 1] = []
    compatible_ppls_set = kwargs.get('compatible_ppls_set', set())

    # Pre-extract typed memoryviews
    cdef double[:] tw_start = customer_arrays['tw_start']
    cdef double[:] tw_end = customer_arrays['tw_end']
    cdef double[:] service_time = customer_arrays['service_time']
    cdef double[:] demand_arr = customer_arrays['demand']
    cdef double[:] frys_arr = customer_arrays['frys']
    cdef double[:] vol_arr = customer_arrays['volume_m3']
    cdef double[:] wt_arr = customer_arrays['weight_kg']
    cdef np.int32_t[:] biltype_arr = customer_arrays['biltype']

    cdef int cust, cust_addr, r_idx, i, n_routes, route_len
    cdef double d, earliest_dep, shift_start_val
    cdef double cap_ppl, cap_frys, cap_vol, cap_wt, veh_ppl
    cdef int prev_addr, next_addr
    cdef list route, feasible_options
    cdef double regret_val
    cdef int n_opts, choice_idx, final_r, final_p
    cdef double[:] cost_view
    cdef int lunch_pos_val
    cdef int phase_trip, is_trip1

    cdef int max_fp = 4096
    cdef double[4096] probs_buf
    cdef int max_trips = 4  # MAX_TRIPS

    for phase_trip in range(1, max_trips + 1):
        if phase_trip > 1:
            _validate_later_trips(new_sol, time_matrix_array, customer_addr_idx,
                                   tw_start, tw_end, service_time, depot_idx)
            unassigned.extend(new_sol.routes[len(new_sol.routes) - 1])
            new_sol.routes[len(new_sol.routes) - 1] = []

        while unassigned:
            potential_insertions = []

            for cust in unassigned:
                cust_addr = customer_addr_idx[cust - 1]
                feasible_options = []

                n_routes = len(new_sol.routes) - 1
                for r_idx in range(n_routes):
                    route = new_sol.routes[r_idx]
                    route_len = len(route)
                    meta = new_sol.route_meta[r_idx]

                    if meta is None or meta['trip'] != phase_trip:
                        continue
                    if phase_trip > 1 and not new_sol.routes[r_idx - 1]:
                        continue

                    vehicle_name = new_sol.vehicles[r_idx]
                    cap = vehicles_dict[vehicle_name]
                    cap_ppl = cap['PPL total']
                    cap_frys = cap['PPL Frys']
                    cap_vol = cap['m3']
                    cap_wt = cap['Vekt (KG)']
                    veh_ppl = cap_ppl

                    if not c_check_capacity(route, route_len, cust, demand_arr, frys_arr,
                                             vol_arr, wt_arr, cap_ppl, cap_frys, cap_vol, cap_wt):
                        continue
                    if not c_check_biltype(route, route_len, cust, biltype_arr,
                                            veh_ppl, compatible_ppls_set):
                        continue

                    earliest_dep = c_get_earliest_departure(
                        new_sol, r_idx, time_matrix_array, customer_addr_idx,
                        tw_start, service_time, depot_idx
                    )
                    shift_start_val = <double>meta['shift_start']
                    max_trip_h = <double>meta['max_trip_hours']
                    is_trip1 = 1 if phase_trip == 1 else 0
                    lunch_buf = LUNCH_DURATION_C if is_trip1 else 0.0

                    for i in range(route_len + 1):
                        prev_addr = depot_idx if i == 0 else customer_addr_idx[route[i - 1] - 1]
                        next_addr = depot_idx if i == route_len else customer_addr_idx[route[i] - 1]

                        d = delta_cost(time_matrix_array, prev_addr, cust_addr, next_addr)

                        if not c_check_tw_with_insertion(
                                route, route_len, i, cust,
                                time_matrix_array, customer_addr_idx,
                                tw_start, tw_end, service_time,
                                depot_idx, earliest_dep):
                            continue

                        # Trip duration check
                        candidate_route = route[:i] + [cust] + route[i:]
                        cand_len = route_len + 1
                        return_time = c_compute_return_time(
                            candidate_route, cand_len, time_matrix_array,
                            customer_addr_idx, tw_start, service_time,
                            depot_idx, earliest_dep, 0, 0.0
                        )
                        if not c_check_shift(return_time, DELOADING_TIME_C,
                                              lunch_buf, earliest_dep + max_trip_h):
                            continue

                        if is_trip1:
                            lunch_pos_val = c_find_lunch_position(
                                candidate_route, cand_len, time_matrix_array,
                                customer_addr_idx, tw_start, tw_end, service_time,
                                depot_idx, LUNCH_DURATION_C, earliest_dep, shift_start_val
                            )
                            if lunch_pos_val == 0:
                                continue

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
                break

            potential_insertions.sort(key=lambda x: x[0], reverse=True)
            best = potential_insertions[0]
            target_cust = best[1]
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

            sel_meta = new_sol.route_meta[final_r]
            if phase_trip == 1:
                sel_dep = c_get_earliest_departure(
                    new_sol, final_r, time_matrix_array, customer_addr_idx,
                    tw_start, service_time, depot_idx
                )
                sel_route = new_sol.routes[final_r]
                lunch_pos_val = c_find_lunch_position(
                    sel_route, len(sel_route), time_matrix_array,
                    customer_addr_idx, tw_start, tw_end, service_time,
                    depot_idx, LUNCH_DURATION_C, sel_dep, <double>sel_meta['shift_start']
                )
                new_sol.lunch_breaks[final_r] = lunch_pos_val if lunch_pos_val > 0 else None
            else:
                new_sol.lunch_breaks[final_r] = None
            unassigned.remove(target_cust)

    new_sol.routes[len(new_sol.routes) - 1] = unassigned
    return new_sol


# =============================================================
#                 LOCAL SEARCH — 2-opt
# =============================================================

def two_opt_local_search(solution,
                         double[:, :] time_matrix_array,
                         np.intp_t[:] customer_addr_idx,
                         customer_arrays,
                         int depot_idx=0, **kwargs):
    """Apply 2-opt local search to every route (multi-trip aware)."""
    new_sol = solution.copy()

    cdef double[:] tw_start = customer_arrays['tw_start']
    cdef double[:] tw_end = customer_arrays['tw_end']
    cdef double[:] service_time = customer_arrays['service_time']

    cdef int r_idx, n, i, j, passes
    cdef double earliest_dep, shift_start_val, max_trip_h, lunch_buf, return_time
    cdef int is_trip1
    cdef double old_edges, new_edges
    cdef int lunch_pos_val
    cdef bint improved
    cdef list route, candidate, route_addrs

    for r_idx in range(len(new_sol.routes) - 1):
        route = new_sol.routes[r_idx]
        n = len(route)
        if n < 2:
            continue

        meta = new_sol.route_meta[r_idx]
        earliest_dep = c_get_earliest_departure(
            new_sol, r_idx, time_matrix_array, customer_addr_idx,
            tw_start, service_time, depot_idx
        )
        shift_start_val = <double>meta['shift_start'] if meta else 6.0
        max_trip_h = <double>meta['max_trip_hours'] if meta else 8.0
        is_trip1 = 1 if (meta is not None and meta['trip'] == 1) else 0
        lunch_buf = LUNCH_DURATION_C if is_trip1 else 0.0

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

                        if not c_check_tw_route(candidate, len(candidate),
                                                 time_matrix_array, customer_addr_idx,
                                                 tw_start, tw_end, service_time,
                                                 depot_idx, earliest_dep):
                            continue

                        # Trip duration check
                        return_time = c_compute_return_time(
                            candidate, len(candidate), time_matrix_array,
                            customer_addr_idx, tw_start, service_time,
                            depot_idx, earliest_dep, 0, 0.0
                        )
                        if not c_check_shift(return_time, DELOADING_TIME_C,
                                              lunch_buf, earliest_dep + max_trip_h):
                            continue

                        if is_trip1:
                            lunch_pos_val = c_find_lunch_position(
                                candidate, len(candidate), time_matrix_array,
                                customer_addr_idx, tw_start, tw_end, service_time,
                                depot_idx, LUNCH_DURATION_C, earliest_dep, shift_start_val
                            )
                            if lunch_pos_val == 0:
                                continue

                        route = candidate
                        route_addrs = [depot_idx] + [customer_addr_idx[c - 1] for c in route] + [depot_idx]
                        improved = True
                        break
                if improved:
                    break

        new_sol.routes[r_idx] = route
        if is_trip1:
            lunch_pos_val = c_find_lunch_position(
                route, len(route), time_matrix_array, customer_addr_idx,
                tw_start, tw_end, service_time,
                depot_idx, LUNCH_DURATION_C, earliest_dep, shift_start_val
            )
            new_sol.lunch_breaks[r_idx] = lunch_pos_val if lunch_pos_val > 0 else None
        else:
            new_sol.lunch_breaks[r_idx] = None

    return new_sol


# =============================================================
#                 LOCAL SEARCH — or-opt
# =============================================================

def or_opt_local_search(solution,
                        double[:, :] time_matrix_array,
                        np.intp_t[:] customer_addr_idx,
                        customer_arrays,
                        int depot_idx=0, **kwargs):
    """Apply or-opt-1 (relocate) local search to every route (multi-trip aware)."""
    new_sol = solution.copy()

    cdef double[:] tw_start = customer_arrays['tw_start']
    cdef double[:] tw_end = customer_arrays['tw_end']
    cdef double[:] service_time = customer_arrays['service_time']

    cdef int r_idx, n, k, p, passes
    cdef double earliest_dep, shift_start_val, max_trip_h, lunch_buf, return_time
    cdef int is_trip1
    cdef double removal_gain, insertion_cost
    cdef int lunch_pos_val
    cdef bint improved
    cdef list route, candidate, route_addrs, reduced_addrs, route_without_k
    cdef int prev_k, curr_k, next_k, prev_p, next_p

    for r_idx in range(len(new_sol.routes) - 1):
        route = new_sol.routes[r_idx]
        if len(route) < 2:
            continue

        meta = new_sol.route_meta[r_idx]
        earliest_dep = c_get_earliest_departure(
            new_sol, r_idx, time_matrix_array, customer_addr_idx,
            tw_start, service_time, depot_idx
        )
        shift_start_val = <double>meta['shift_start'] if meta else 6.0
        max_trip_h = <double>meta['max_trip_hours'] if meta else 8.0
        is_trip1 = 1 if (meta is not None and meta['trip'] == 1) else 0
        lunch_buf = LUNCH_DURATION_C if is_trip1 else 0.0

        improved = True
        passes = 0
        while improved and passes < 10:
            improved = False
            passes += 1
            n = len(route)
            route_addrs = [depot_idx] + [customer_addr_idx[c - 1] for c in route] + [depot_idx]

            for k in range(1, n + 1):
                prev_k = route_addrs[k - 1]
                curr_k = route_addrs[k]
                next_k = route_addrs[k + 1]
                removal_gain = (time_matrix_array[prev_k, curr_k] +
                                time_matrix_array[curr_k, next_k] -
                                time_matrix_array[prev_k, next_k])

                reduced_addrs = route_addrs[:k] + route_addrs[k + 1:]

                for p in range(1, n + 1):
                    if p == k:
                        continue
                    prev_p = reduced_addrs[p - 1]
                    next_p = reduced_addrs[p]
                    insertion_cost = (time_matrix_array[prev_p, curr_k] +
                                      time_matrix_array[curr_k, next_p] -
                                      time_matrix_array[prev_p, next_p])

                    if removal_gain - insertion_cost > 1e-6:
                        route_without_k = route[:k - 1] + route[k:]
                        candidate = route_without_k[:p - 1] + [route[k - 1]] + route_without_k[p - 1:]

                        if not c_check_tw_route(candidate, len(candidate),
                                                 time_matrix_array, customer_addr_idx,
                                                 tw_start, tw_end, service_time,
                                                 depot_idx, earliest_dep):
                            continue

                        # Trip duration check
                        return_time = c_compute_return_time(
                            candidate, len(candidate), time_matrix_array,
                            customer_addr_idx, tw_start, service_time,
                            depot_idx, earliest_dep, 0, 0.0
                        )
                        if not c_check_shift(return_time, DELOADING_TIME_C,
                                              lunch_buf, earliest_dep + max_trip_h):
                            continue

                        if is_trip1:
                            lunch_pos_val = c_find_lunch_position(
                                candidate, len(candidate), time_matrix_array,
                                customer_addr_idx, tw_start, tw_end, service_time,
                                depot_idx, LUNCH_DURATION_C, earliest_dep, shift_start_val
                            )
                            if lunch_pos_val == 0:
                                continue

                        route = candidate
                        improved = True
                        break
                if improved:
                    break

        new_sol.routes[r_idx] = route
        if is_trip1:
            lunch_pos_val = c_find_lunch_position(
                route, len(route), time_matrix_array, customer_addr_idx,
                tw_start, tw_end, service_time,
                depot_idx, LUNCH_DURATION_C, earliest_dep, shift_start_val
            )
            new_sol.lunch_breaks[r_idx] = lunch_pos_val if lunch_pos_val > 0 else None
        else:
            new_sol.lunch_breaks[r_idx] = None

    return new_sol
