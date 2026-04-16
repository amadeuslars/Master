# feasibilitycheck.py
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cost import calculate_route_cost

# Multi-trip timing constants (hours)
LOADING_TIME = 1.0        # Time to load vehicle at depot before departure
DELOADING_TIME = 1/3      # Time to unload at depot after return (20 min)
LUNCH_DURATION = 0.5      # Lunch break duration
LUNCH_DEADLINE_HOURS = 6.0  # Lunch must occur within first 6h of shift


# ---------------------------------------------------------
#  Multi-trip helpers
# ---------------------------------------------------------

def get_earliest_departure(solution, r_idx, time_matrix_array, customer_addr_idx,
                           customer_arrays, depot_idx):
    """
    Get the earliest time loading can start for this trip slot.
    For trip 1: shift_start.
    For trip N>1: end of trip N-1 (return + deload) on same vehicle.
    """
    meta = solution.route_meta[r_idx]
    if meta is None:
        return 0.0

    if meta['trip'] == 1:
        return meta['shift_start']

    # Trip N>1: find previous trip on same vehicle (always the preceding slot)
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
        lunch_pos = solution.lunch_breaks[prev_idx] if hasattr(solution, 'lunch_breaks') else None
        return_time = compute_trip_return_time(
            prev_route, time_matrix_array, customer_addr_idx,
            customer_arrays, depot_idx, prev_departure,
            lunch_pos=lunch_pos
        )
        return return_time + DELOADING_TIME

    return meta['shift_start']


def compute_trip_return_time(route, time_matrix_array, customer_addr_idx,
                             customer_arrays, depot_idx, earliest_departure,
                             lunch_pos=None, lunch_duration=LUNCH_DURATION):
    """
    Compute when a vehicle returns to depot after completing a trip.
    Uses per-customer service times from customer_arrays.
    Optionally includes lunch break at lunch_pos (after k-th customer).
    """
    if not route:
        return earliest_departure

    depart_depot = earliest_departure + LOADING_TIME
    current_time = depart_depot
    last_addr = depot_idx

    for i, cust in enumerate(route):
        curr_addr = customer_addr_idx[cust - 1]
        current_time += time_matrix_array[last_addr, curr_addr]
        tw_start = float(customer_arrays['tw_start'][cust - 1])
        current_time = max(current_time, tw_start)
        service = float(customer_arrays['service_time'][cust - 1])
        current_time += service
        last_addr = curr_addr
        # Lunch break after this customer
        if lunch_pos is not None and lunch_pos == i + 1:
            current_time += lunch_duration

    # Return to depot
    current_time += time_matrix_array[last_addr, depot_idx]
    return current_time


def check_shift_feasibility(route, time_matrix_array, customer_addr_idx,
                            customer_arrays, depot_idx, earliest_departure,
                            shift_end, lunch_duration=0.0, debug=False):
    """Legacy shift check — kept for backward compatibility."""
    if not route:
        return True
    return_time = compute_trip_return_time(
        route, time_matrix_array, customer_addr_idx,
        customer_arrays, depot_idx, earliest_departure
    )
    end_time = return_time + DELOADING_TIME + lunch_duration
    return end_time <= shift_end + 0.01


def check_trip_duration_feasibility(route, time_matrix_array, customer_addr_idx,
                                    customer_arrays, depot_idx, earliest_departure,
                                    max_trip_hours, lunch_duration=0.0):
    """
    Check that a single trip fits within the max trip duration.
    Trip duration = (return + deload + lunch) - earliest_departure.
    """
    if not route:
        return True
    return_time = compute_trip_return_time(
        route, time_matrix_array, customer_addr_idx,
        customer_arrays, depot_idx, earliest_departure
    )
    trip_end = return_time + DELOADING_TIME + lunch_duration
    return trip_end <= earliest_departure + max_trip_hours + 0.01


# ---------------------------------------------------------
#  Existing feasibility checks (updated for multi-trip)
# ---------------------------------------------------------

def check_capacity_feasibility(route_indices, vehicle_name, vehicles_dict, customer_arrays, debug=False):
    """
    Fast capacity check using pre-fetched vehicle dict and numpy arrays.
    route_indices: list of 1-based customer IDs. Arrays are 0-based (use c-1).
    """
    if not route_indices:
        return True

    try:
        cap = vehicles_dict[vehicle_name]
        idx = np.array([c - 1 for c in route_indices], dtype=np.int32)
        ppl_tol = 1/3 - 1e-6  # under 1/3 pallet overshoot is rounding noise
        tol = 1e-6             # float tolerance for volume/weight

        if float(customer_arrays['demand'][idx].sum()) > cap['PPL total'] + ppl_tol: return False
        if float(customer_arrays['frys'][idx].sum()) > cap['PPL Frys'] + ppl_tol: return False
        if float(customer_arrays['volume_m3'][idx].sum()) > cap['m3'] + tol: return False
        if float(customer_arrays['weight_kg'][idx].sum()) > cap['Vekt (KG)'] + tol: return False

        return True

    except KeyError:
        if debug: print(f"Vehicle {vehicle_name} not found")
        return False

def check_time_window_feasibility(
    solution_indices,
    time_matrix_array,
    customer_addr_idx,
    customer_arrays,
    depot_idx,
    earliest_departure=None,
    debug=False,
):
    """
    Fast time window check with optional shift-aware departure time.
    When earliest_departure is set, vehicle departs depot at earliest_departure + LOADING_TIME.
    When None, uses legacy behavior (optimized departure to minimize waiting).
    """
    if not solution_indices:
        return True

    first_idx = solution_indices[0]
    first_addr_idx = customer_addr_idx[first_idx - 1]
    travel_to_first = time_matrix_array[depot_idx, first_addr_idx]
    first_tw_start = float(customer_arrays['tw_start'][first_idx - 1])

    if earliest_departure is not None:
        depart_time = earliest_departure + LOADING_TIME
        current_time = depart_time + travel_to_first
    else:
        current_time = max(0.0, first_tw_start - travel_to_first)
        current_time += travel_to_first

    if current_time > float(customer_arrays['tw_end'][first_idx - 1]):
        return False

    current_time = max(current_time, first_tw_start) + float(customer_arrays['service_time'][first_idx - 1])
    last_idx = first_addr_idx

    for i in range(1, len(solution_indices)):
        cust_idx = solution_indices[i]
        curr_addr_idx = customer_addr_idx[cust_idx - 1]

        travel = time_matrix_array[last_idx, curr_addr_idx]
        arrival = current_time + travel

        tw_start = float(customer_arrays['tw_start'][cust_idx - 1])
        tw_end = float(customer_arrays['tw_end'][cust_idx - 1])

        if arrival > tw_end:
            if debug: print(f"Late at {cust_idx}: {arrival} > {tw_end}")
            return False

        current_time = max(arrival, tw_start) + float(customer_arrays['service_time'][cust_idx - 1])
        last_idx = curr_addr_idx

    return True

def check_max_route_duration(route_indices, customer_addr_idx, time_matrix_array, depot_idx,
                             max_hours=8.0, lunch_break=0.5, service_time_per_customer=1/3,
                             loading_time=1.0, deloading_time=1/3, debug=False):
    """
    Check that total route duration does not exceed max_hours.
    Total duration = loading + driving (depot->customers->depot) + service + lunch + deloading.
    All times in hours. Time matrix must already be in hours.
    NOTE: Kept for backward compatibility. Multi-trip uses check_trip_duration_feasibility.
    """
    if not route_indices:
        return True

    # Driving time: depot -> first customer -> ... -> last customer -> depot
    driving_time = 0.0
    last_idx = depot_idx
    for cust in route_indices:
        curr_idx = customer_addr_idx[cust - 1]
        driving_time += time_matrix_array[last_idx, curr_idx]
        last_idx = curr_idx
    driving_time += time_matrix_array[last_idx, depot_idx]

    total_service = service_time_per_customer * len(route_indices)
    total_duration = loading_time + driving_time + lunch_break + total_service + deloading_time

    if debug:
        print(f"Route duration: load={loading_time:.2f}h + drive={driving_time:.2f}h + lunch={lunch_break:.2f}h + service={total_service:.2f}h + deload={deloading_time:.2f}h = {total_duration:.2f}h (max {max_hours}h)")

    return total_duration <= max_hours


def find_lunch_break_position(route_indices, time_matrix_array, customer_addr_idx,
                               customer_arrays, depot_idx, lunch_duration=0.5,
                               earliest_departure=None, shift_start=None):
    """
    Find the earliest feasible position for a lunch break in the route.
    Positions 1..n: after serving the k-th customer (no break before first customer).
    Returns the position (int) or None if no valid placement exists.
    """
    if not route_indices:
        return None

    n = len(route_indices)
    for break_pos in range(1, n + 1):
        if _simulate_route_with_break(route_indices, break_pos, time_matrix_array,
                                      customer_addr_idx, customer_arrays, depot_idx,
                                      lunch_duration,
                                      earliest_departure, shift_start):
            return break_pos
    return None


def check_lunch_break_feasibility(route_indices, time_matrix_array, customer_addr_idx,
                                   customer_arrays, depot_idx, lunch_duration=0.5,
                                   earliest_departure=None, shift_start=None,
                                   debug=False):
    """
    Check that a lunch break can be placed somewhere in the route (after at least one customer)
    without violating any time windows. Returns True if a valid placement exists.
    """
    if not route_indices:
        return True

    result = find_lunch_break_position(route_indices, time_matrix_array, customer_addr_idx,
                                        customer_arrays, depot_idx, lunch_duration,
                                        earliest_departure, shift_start)
    if result is None and debug:
        print(f"No valid lunch break position for route {route_indices}")
    return result is not None


def _simulate_route_with_break(route_indices, break_pos, time_matrix_array,
                                customer_addr_idx, customer_arrays, depot_idx,
                                lunch_duration,
                                earliest_departure=None, shift_start=None):
    """
    Simulate route timing with a lunch break inserted at break_pos.
    break_pos=k (1..n): break after serving the k-th customer in the route.
    Uses absolute time when earliest_departure is provided.
    Uses per-customer service times from customer_arrays.
    """
    if earliest_departure is not None:
        current_time = earliest_departure + LOADING_TIME
    else:
        current_time = 0.0
    last_addr = depot_idx

    for i, cust in enumerate(route_indices):
        curr_addr = customer_addr_idx[cust - 1]
        current_time += time_matrix_array[last_addr, curr_addr]

        tw_end = float(customer_arrays['tw_end'][cust - 1])
        if current_time > tw_end:
            return False

        tw_start = float(customer_arrays['tw_start'][cust - 1])
        service = float(customer_arrays['service_time'][cust - 1])
        current_time = max(current_time, tw_start) + service
        last_addr = curr_addr

        if break_pos == i + 1:
            # Validate lunch falls within shift deadline
            if shift_start is not None:
                lunch_time = current_time  # lunch starts after serving this customer
                if lunch_time > shift_start + LUNCH_DEADLINE_HOURS:
                    return False
            current_time += lunch_duration

    return True


def compute_route_schedule(route, time_matrix_array, customer_addr_idx,
                           customer_arrays, depot_idx, earliest_departure,
                           lunch_pos=None, lunch_duration=LUNCH_DURATION):
    """
    Build a detailed timeline for a single trip.
    Returns a list of event dicts in chronological order:
      {'event': str, 'time': float, 'customer': int|None, 'details': str}
    """
    events = []
    if not route:
        return events

    load_start = earliest_departure
    load_end = load_start + LOADING_TIME
    events.append({'event': 'load_start', 'time': load_start, 'customer': None, 'details': 'Loading at depot'})
    events.append({'event': 'depart_depot', 'time': load_end, 'customer': None, 'details': 'Depart depot'})

    current_time = load_end
    last_addr = depot_idx

    for i, cust in enumerate(route):
        curr_addr = customer_addr_idx[cust - 1]
        travel = time_matrix_array[last_addr, curr_addr]
        arrive = current_time + travel
        tw_start = float(customer_arrays['tw_start'][cust - 1])
        tw_end = float(customer_arrays['tw_end'][cust - 1])
        service = float(customer_arrays['service_time'][cust - 1])
        wait = max(0.0, tw_start - arrive)
        serve_start = max(arrive, tw_start)
        serve_end = serve_start + service

        events.append({
            'event': 'arrive', 'time': arrive, 'customer': cust,
            'details': f'Arrive (TW {tw_start:.2f}-{tw_end:.2f}, wait {wait*60:.0f}min)'
        })
        events.append({
            'event': 'serve_end', 'time': serve_end, 'customer': cust,
            'details': f'Serve done ({service*60:.0f}min)'
        })

        current_time = serve_end
        last_addr = curr_addr

        # Lunch break after this customer
        if lunch_pos is not None and lunch_pos == i + 1:
            events.append({
                'event': 'lunch', 'time': current_time, 'customer': None,
                'details': f'Lunch break ({lunch_duration*60:.0f}min)'
            })
            current_time += lunch_duration

    # Return to depot
    travel_back = time_matrix_array[last_addr, depot_idx]
    return_time = current_time + travel_back
    events.append({'event': 'return_depot', 'time': return_time, 'customer': None,
                   'details': f'Return to depot (drove {travel_back*60:.0f}min)'})

    deload_end = return_time + DELOADING_TIME
    events.append({'event': 'deload_end', 'time': deload_end, 'customer': None,
                   'details': 'Deloading done'})

    return events


def check_vehicle_store_compatibility(route_indices, vehicle_name, vehicles_dict, customer_arrays, compatible_ppls_set):
    """
    Checks compatibility using integer arrays.

    compatible_ppls_set: set of PPL capacities allowed for 'Biltype 2'
    """
    if not route_indices:
        return True

    idx = np.array([c - 1 for c in route_indices], dtype=np.int32)
    biltypes = customer_arrays['biltype'][idx]

    if np.all(biltypes == 1):
        return True

    if np.any(biltypes == 2):
        veh_ppl = vehicles_dict[vehicle_name]['PPL total']
        if veh_ppl not in compatible_ppls_set:
            return False

    if np.any(biltypes > 2):
        return False

    return True


# ---------------------------------------------------------
#  Post-repair validation and trimming
# ---------------------------------------------------------

def validate_and_trim_routes(solution, time_matrix_array, customer_addr_idx,
                             customer_arrays, depot_idx, verbose=False):
    """
    Safety-net pass: check every non-dummy route for max trip duration using
    the ACTUAL lunch position. If a route overflows, remove customers from the
    end (one at a time) until it fits. Removed customers go to the dummy route.

    Call this after every repair operator to guarantee trip duration feasibility.
    Returns the number of customers moved to dummy.
    """
    moved = 0
    for r_idx in range(len(solution.routes) - 1):
        route = solution.routes[r_idx]
        if not route:
            continue
        meta = solution.route_meta[r_idx]
        if meta is None:
            continue

        max_trip_h = meta.get('max_trip_hours', 8.0)
        is_trip1 = meta['trip'] == 1

        earliest_dep = get_earliest_departure(
            solution, r_idx, time_matrix_array, customer_addr_idx,
            customer_arrays, depot_idx
        )

        # Use actual lunch position for trip-1
        lunch_pos = solution.lunch_breaks[r_idx] if is_trip1 else None
        lunch_dur = LUNCH_DURATION if is_trip1 else 0.0

        return_time = compute_trip_return_time(
            route, time_matrix_array, customer_addr_idx,
            customer_arrays, depot_idx, earliest_dep,
            lunch_pos=lunch_pos
        )
        end_time = return_time + DELOADING_TIME

        while route and end_time > earliest_dep + max_trip_h + 0.01:
            removed_cust = route.pop()
            solution.routes[-1].append(removed_cust)
            moved += 1
            if verbose:
                print(f"  [trim] Removed cust {removed_cust} from route {r_idx} "
                      f"(end={end_time:.3f}h > max_end={earliest_dep + max_trip_h:.3f}h)")
            if not route:
                solution.lunch_breaks[r_idx] = None
                break

            # Recompute lunch position and return time
            if is_trip1:
                lunch_pos = find_lunch_break_position(
                    route, time_matrix_array, customer_addr_idx,
                    customer_arrays, depot_idx,
                    earliest_departure=earliest_dep,
                    shift_start=meta['shift_start']
                )
                solution.lunch_breaks[r_idx] = lunch_pos
            else:
                lunch_pos = None

            return_time = compute_trip_return_time(
                route, time_matrix_array, customer_addr_idx,
                customer_arrays, depot_idx, earliest_dep,
                lunch_pos=lunch_pos
            )
            end_time = return_time + DELOADING_TIME

    return moved
