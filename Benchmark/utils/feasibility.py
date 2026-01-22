import pandas as pd
import numpy as np
import os

def check_capacity_feasibility(route_indices, vehicle_name, vehicles_df, customer_arrays):
    """
    Fast capacity check using vehicle DataFrame and numpy arrays.
    For Solomon instances: only checks demand vs capacity.
    """
    if not route_indices:
        return True

    try:
        # Get vehicle capacity
        vehicle_capacity = vehicles_df.loc[vehicle_name, 'capacity']
        
        # Convert route_indices to numpy array
        idx = np.asarray(route_indices, dtype=np.int32)
        
        # Sum demands for all customers in route
        total_demand = float(customer_arrays['demand'][idx-1].sum())
        
        if total_demand > vehicle_capacity:
            return False
            
        return True

    except KeyError:
        print(f"Vehicle {vehicle_name} not found")
        return False

def check_time_window_feasibility(
    route_indices,
    distance_matrix_array,
    customer_addr_idx,
    customer_arrays,
    depot_idx=0,
    debug=False,
):
    """
    Fast time window check for Solomon instances.
    Travel time = distance (1 distance unit = 1 minute).
    """
    if not route_indices:
        return True

    current_time = 0.0
    current_location = depot_idx  # Start at depot
    
    for cust_idx in route_indices:
        # Get customer's position in distance matrix
        cust_addr_idx = customer_addr_idx[cust_idx-1]
        
        # Travel time equals distance
        travel_time = distance_matrix_array[current_location, cust_addr_idx]
        arrival_time = current_time + travel_time
        
        # Get time window
        tw_start = float(customer_arrays['tw_start'][cust_idx-1])
        tw_end = float(customer_arrays['tw_end'][cust_idx-1])
        
        # Check if we arrive too late
        if arrival_time > tw_end:
            if debug:
                print(f"Late at customer {cust_idx}: arrival={arrival_time:.2f} > due_date={tw_end:.2f}")
            return False
        
        # Service starts at max(arrival_time, tw_start)
        service_start = max(arrival_time, tw_start)
        service_time = float(customer_arrays['service_time'][cust_idx-1])
        
        # Update current time and location
        current_time = service_start + service_time
        current_location = cust_addr_idx
    
    return True

