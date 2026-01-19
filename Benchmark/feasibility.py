import pandas as pd
import numpy as np
import os

def load_vrp_data(customers_file='c1_6_1/customers.csv', vehicles_file='c1_6_1/vehicles.csv', distance_matrix_file='c1_6_1/distance_matrix.csv'):
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Make paths relative to script directory
    customers_file = os.path.join(script_dir, customers_file)
    vehicles_file = os.path.join(script_dir, vehicles_file)
    distance_matrix_file = os.path.join(script_dir, distance_matrix_file)
    
    customers_df = pd.read_csv(customers_file)
    vehicles_df = pd.read_csv(vehicles_file)
    distance_matrix_df = pd.read_csv(distance_matrix_file, index_col=0)
    distance_matrix_df = distance_matrix_df.apply(pd.to_numeric, errors='coerce')

    vehicles_df.set_index('vehicle_type', inplace=True)

    # Keep time windows in minutes (Solomon standard format)
    # No conversion needed - travel time = distance (1 distance unit = 1 minute)
    
    distance_matrix_array = distance_matrix_df.to_numpy(dtype=np.float32)
    
    # Map customer IDs to matrix indices (depot is index 0, customers start at 1)
    customer_addr_idx = customers_df['customer_id'].astype(np.int32).to_numpy()

    # Create compact arrays for fast checking
    def _num(col, default=0.0, scale=1.0, dtype=np.float32):
        vals = customers_df[col].fillna(default)
        if scale != 1.0:
            vals = vals * scale
        return vals.to_numpy(dtype)

    customer_arrays = {
        'demand': _num('demand', 0.0, 1.0, np.float32),
        'tw_start': _num('ready_time', 0.0, 1.0, np.float32),  # Keep in minutes
        'tw_end': _num('due_date', 0.0, 1.0, np.float32),      # Keep in minutes
        'service_time': _num('service_time', 0.0, 1.0, np.float32),  # Keep in minutes
    }

    return customers_df, vehicles_df, distance_matrix_df, distance_matrix_array, customer_addr_idx, customer_arrays


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

