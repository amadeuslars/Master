# feasibilitycheck.py
import pandas as pd
import numpy as np

def load_vrp_data(customers_file='customers.csv', vehicles_file='vehicles.csv', time_matrix_file='time_matrix.csv'):
    customers_df = pd.read_csv(customers_file)
    vehicles_df = pd.read_csv(vehicles_file)
    time_matrix_df = pd.read_csv(time_matrix_file, index_col=0)
    time_matrix_df = time_matrix_df.apply(pd.to_numeric, errors='coerce') / 3600.0

    vehicles_df.set_index('Navn', inplace=True)

    def time_str_to_hours(t_str):
        try:
            return pd.to_timedelta(str(t_str) + ':00').total_seconds() / 3600.0
        except:
            return 0.0

    customers_df['tw_start_hours'] = customers_df['Leveringstid kunde fra'].apply(time_str_to_hours)
    customers_df['tw_end_hours'] = customers_df['Leveringstid kunde til'].apply(time_str_to_hours)
    
    time_matrix_array = time_matrix_df.to_numpy(dtype=np.float32)
    address_to_idx = {addr: i for i, addr in enumerate(time_matrix_df.index)}

    if 'Adresse' not in customers_df.columns:
        raise KeyError("Expected column 'Adresse' in customers.csv")
    customers_df['addr_idx'] = customers_df['Adresse'].map(address_to_idx)
    customer_addr_idx = customers_df['addr_idx'].astype(np.int32).to_numpy()

    # Create compact arrays for fast checking
    def _num(col, default=0.0, scale=1.0, dtype=np.float32):
        vals = customers_df[col].fillna(default)
        if scale != 1.0:
            vals = vals * scale
        return vals.to_numpy(dtype)

    customer_arrays = {
        'pallets': _num('PPL pr leveranse', 0.0, 1.0, np.float32),
        'volume_m3': _num('Volum pr leveranse (m3)', 0.0, 1.0, np.float32),
        'weight_kg': _num('Vekt pr leveranse (tonn)', 0.0, 1000.0, np.float32),
        'frys': _num('PPL hvorav frys', 0.0, 1.0, np.float32),
        'tw_start': customers_df['tw_start_hours'].fillna(0.0).to_numpy(np.float32),
        'tw_end': customers_df['tw_end_hours'].fillna(0.0).to_numpy(np.float32),
        'biltype': customers_df['Biltype'].fillna(0).astype(np.int16).to_numpy(),
    }

    return customers_df, vehicles_df, time_matrix_df, time_matrix_array, address_to_idx, customer_addr_idx, customer_arrays

def check_capacity_feasibility(route_indices, vehicle_name, vehicles_dict, customer_arrays, debug=False):
    """
    Fast capacity check using pre-fetched vehicle dict and numpy arrays.
    """
    if not route_indices:
        return True

    try:
        # Get vehicle specs from dict (O(1) lookup)
        cap = vehicles_dict[vehicle_name]
        
        idx = np.asarray(route_indices, dtype=np.int32)
        
        # Vectorized sums
        total_pallets = float(customer_arrays['pallets'][idx].sum())
        if total_pallets > cap['PPL total']: return False
        
        total_volume = float(customer_arrays['volume_m3'][idx].sum())
        if total_volume > cap['m3']: return False
        
        total_weight_kg = float(customer_arrays['weight_kg'][idx].sum())
        if total_weight_kg > cap['Vekt (KG)']: return False
        
        total_frys = float(customer_arrays['frys'][idx].sum())
        if total_frys > cap['PPL Frys']: return False
            
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
    # Travel from Depot to First
    first_addr_idx = customer_addr_idx[first_idx]
    travel_to_first = time_matrix_array[depot_idx, first_addr_idx]

    first_tw_start = float(customer_arrays['tw_start'][first_idx])
    
    # Start time logic
    current_time = max(0.0, first_tw_start - travel_to_first)
    current_time += travel_to_first # Arrive at first
    
    # Check first node window immediately
    if current_time > float(customer_arrays['tw_end'][first_idx]):
        return False
        
    current_time = max(current_time, first_tw_start) + service_time
    last_idx = first_addr_idx

    # Iterate rest
    for i in range(1, len(solution_indices)):
        cust_idx = solution_indices[i]
        curr_addr_idx = customer_addr_idx[cust_idx]
        
        travel = time_matrix_array[last_idx, curr_addr_idx]
        arrival = current_time + travel
        
        tw_start = float(customer_arrays['tw_start'][cust_idx])
        tw_end = float(customer_arrays['tw_end'][cust_idx])

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
        
    # Get Biltypes for all customers in route at once
    biltypes = customer_arrays['biltype'][route_indices]
    
    # If all are Type 1 (universal), we are good
    if np.all(biltypes == 1):
        return True
        
    # If any are Type 2, check vehicle PPL
    if np.any(biltypes == 2):
        veh_ppl = vehicles_dict[vehicle_name]['PPL total']
        if veh_ppl not in compatible_ppls_set:
            return False
            
    # If there are other types (not 1 or 2), assumed incompatible
    # (Checking for values > 2)
    if np.any(biltypes > 2):
        return False
        
    return True