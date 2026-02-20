# feasibilitycheck.py
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cost import calculate_route_cost


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

        if float(customer_arrays['pallets'][idx].sum()) > cap['PPL total']: return False
        if float(customer_arrays['volume_m3'][idx].sum()) > cap['m3']: return False
        if float(customer_arrays['weight_kg'][idx].sum()) > cap['Vekt (KG)']: return False
        if float(customer_arrays['frys'][idx].sum()) > cap['PPL Frys']: return False

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
    service_time=0.5,
    debug=False,
):
    """
    Fast time window check. Logic only, no DataFrame lookups.
    """
    if not solution_indices:
        return True

    first_idx = solution_indices[0]
    first_addr_idx = customer_addr_idx[first_idx - 1]
    travel_to_first = time_matrix_array[depot_idx, first_addr_idx]

    first_tw_start = float(customer_arrays['tw_start'][first_idx - 1])

    current_time = max(0.0, first_tw_start - travel_to_first)
    current_time += travel_to_first

    if current_time > float(customer_arrays['tw_end'][first_idx - 1]):
        return False

    current_time = max(current_time, first_tw_start) + service_time
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

        current_time = max(arrival, tw_start) + service_time
        last_idx = curr_addr_idx

    return True

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

