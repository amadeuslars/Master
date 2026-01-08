import pandas as pd
import numpy as np

def load_vrp_data(customers_file='customers.csv', vehicles_file='vehicles.csv', time_matrix_file='time_matrix.csv'):
    
    customers_df = pd.read_csv(customers_file)
    vehicles_df = pd.read_csv(vehicles_file)
    time_matrix_df = pd.read_csv(time_matrix_file, index_col=0)
    time_matrix_df = time_matrix_df.apply(pd.to_numeric, errors='coerce') / 3600.0

    # Index as 'Navn'
    vehicles_df.set_index('Navn', inplace=True)

    def time_str_to_hours(t_str):
        try:
            return pd.to_timedelta(str(t_str) + ':00').total_seconds() / 3600.0
        except:
            return 0.0 # Default or error value

    # Create new numeric columns for faster checking later
    customers_df['tw_start_hours'] = customers_df['Leveringstid kunde fra'].apply(time_str_to_hours)
    customers_df['tw_end_hours'] = customers_df['Leveringstid kunde til'].apply(time_str_to_hours)
    
    # Optimize time_matrix: convert to NumPy array + address-to-index mapping
    time_matrix_array = time_matrix_df.to_numpy(dtype=np.float32)
    address_to_idx = {addr: i for i, addr in enumerate(time_matrix_df.index)}

    # Pre-index each customer once to the time-matrix row/col index
    if 'Adresse' not in customers_df.columns:
        raise KeyError("Expected column 'Adresse' in customers.csv")
    customers_df['addr_idx'] = customers_df['Adresse'].map(address_to_idx)
    if customers_df['addr_idx'].isna().any():
        missing = customers_df.loc[customers_df['addr_idx'].isna(), 'Adresse'].unique()
        raise ValueError(f"Addresses missing from time_matrix: {missing}")
    customer_addr_idx = customers_df['addr_idx'].astype(np.int32).to_numpy()

    # Extract frequently-used customer columns as compact arrays
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
    
def check_capacity_feasibility(route_customers, vehicle_name, vehicles_df, customer_arrays=None):
    """
    Checks if the total demand of a route exceeds the capacity of the assigned vehicle.

    Args:
        route_customers_df (pd.DataFrame): A DataFrame containing the customer data for a specific route.
        vehicle_name (str): The name of the vehicle (e.g., 'small', 'medium').
        vehicles_df (pd.DataFrame): The DataFrame containing vehicle capacity information.

    Returns:
        bool: True if the route is feasible in terms of capacity, False otherwise.
    """
    # Handle empty route
    if (isinstance(route_customers, (list, tuple)) and len(route_customers) == 0) or (
        hasattr(route_customers, 'empty') and getattr(route_customers, 'empty', False)
    ):
        return True

    try:
        # Get the capacity constraints for the given vehicle name
        vehicle_capacities = vehicles_df.loc[vehicle_name]
        
        # Sum the demands: prefer fast array path when route is indices and arrays provided
        if customer_arrays is not None and isinstance(route_customers, (list, tuple, np.ndarray)):
            idx = np.asarray(route_customers, dtype=np.int32)
            total_pallets = float(customer_arrays['pallets'][idx].sum())
            total_volume = float(customer_arrays['volume_m3'][idx].sum())
            total_weight_kg = float(customer_arrays['weight_kg'][idx].sum())
            total_frys = float(customer_arrays['frys'][idx].sum())
        else:
            # Fallback: route provided as DataFrame
            route_df = route_customers
            total_pallets = route_df['PPL pr leveranse'].sum()
            total_volume = route_df['Volum pr leveranse (m3)'].sum()
            total_weight_kg = route_df['Vekt pr leveranse (tonn)'].sum() * 1000
            total_frys = route_df['PPL hvorav frys'].sum()

        print(f"\n--- Checking Feasibility for Vehicle '{vehicle_name}' ---")
        print(f"Route Demands: Pallets={total_pallets:.2f}, Volume={total_volume:.2f} m3, Weight={total_weight_kg:.2f} KG, Frys={total_frys:.2f} PPL")
        print(f"Vehicle Limits: Pallets={vehicle_capacities['PPL total']:.2f}, Volume={vehicle_capacities['m3']:.2f} m3, Weight={vehicle_capacities['Vekt (KG)']:.2f} KG, Frys={vehicle_capacities['PPL Frys']:.2f} PPL")

        # Check if any demand exceeds the vehicle's capacity
        if total_pallets > vehicle_capacities['PPL total']:
            print("Result: Infeasible - Pallet capacity exceeded.")
            return False
        if total_volume > vehicle_capacities['m3']:
            print("Result: Infeasible - Volume capacity exceeded.")
            return False
        if total_weight_kg > vehicle_capacities['Vekt (KG)']:
            print("Result: Infeasible - Weight capacity exceeded.")
            return False
        if total_frys > vehicle_capacities['PPL Frys']:
            print("Result: Infeasible - Freezer pallet capacity exceeded.")
            return False
            
        print("Result: Feasible")
        return True

    except KeyError:
        print(f"Error: Vehicle name '{vehicle_name}' not found in the vehicle data.")
        return False
    except Exception as e:
        print(f"An error occurred during feasibility check: {e}")
        return False

def check_time_window_feasibility(
    solution_indices,
    customers_df,
    time_matrix_array,
    address_to_idx,
    customer_addr_idx,
    customer_arrays=None,
    service_time=0.5,
    debug=False,
):
    """
    Checks if a route is feasible regarding customer time windows.

    Args:
        solution_indices (list): Indices of customers in visit order.
        customers_df (pd.DataFrame): Customers DataFrame (used for debug prints only).
        time_matrix_array (np.ndarray): Travel times (hours) as dense array.
        address_to_idx (dict): Mapping from address string to matrix index.
        customer_addr_idx (np.ndarray): Per-customer time-matrix indices.
        customer_arrays (dict): Optional arrays with 'tw_start' and 'tw_end' for fast access.
        service_time (float): Time spent at each stop (hours).
        debug (bool): When True, prints detailed steps.

    Returns:
        bool: True if the route is feasible, False otherwise.
    """
    if not solution_indices:
        return True

    first_idx = solution_indices[0]
    try:
        depot_idx = address_to_idx['Depot']
        first_addr_idx = customer_addr_idx[first_idx]
        travel_to_first = time_matrix_array[depot_idx, first_addr_idx]
    except KeyError:
        return False

    # Get first time window start quickly
    if customer_arrays is not None:
        first_tw_start = float(customer_arrays['tw_start'][first_idx])
    else:
        first_tw_start = float(customers_df.iloc[first_idx]['tw_start_hours'])

    # Earliest feasible start time
    start_time = max(0.0, first_tw_start - travel_to_first)
    current_time = start_time
    last_idx = depot_idx
    last_location_name = 'Depot'

    if debug:
        print(f"\n--- Checking Time Window Feasibility ---")

    for customer_index in solution_indices:
        # Travel
        current_idx = customer_addr_idx[customer_index]
        travel_time_hours = time_matrix_array[last_idx, current_idx]

        # Arrival
        arrival_time = current_time + travel_time_hours

        # Window values
        if customer_arrays is not None:
            tw_start = float(customer_arrays['tw_start'][customer_index])
            tw_end = float(customer_arrays['tw_end'][customer_index])
        else:
            row = customers_df.iloc[customer_index]
            tw_start = float(row['tw_start_hours'])
            tw_end = float(row['tw_end_hours'])

        if debug:
            customer_address = customers_df.iloc[customer_index]['Adresse']
            print(f"Visiting '{customer_address}':")
            print(f"  Travel from '{last_location_name}' took {travel_time_hours:.2f}h.")
            print(f"  Arrival Time: {arrival_time:.2f}, Time Window: [{tw_start:.2f} - {tw_end:.2f}]")

        # Window handling
        if arrival_time < tw_start:
            current_time = tw_start
        elif arrival_time > tw_end:
            if debug:
                print(f"  Result: Infeasible - Arrival time {arrival_time:.2f} is after time window closes at {tw_end:.2f}.")
            return False
        else:
            current_time = arrival_time

        # Service time
        current_time += service_time
        if debug:
            print(f"  Departure Time: {current_time:.2f}")

        # Advance
        last_idx = current_idx
        if debug:
            last_location_name = customers_df.iloc[customer_index]['Adresse']

    if debug:
        print("Result: Feasible")
    return True

def check_vehicle_store_compatibility(route_customers_df, vehicle_name, vehicles_df):
    """
    Checks if the assigned vehicle is compatible with all stores on the route.

    Args:
        route_customers_df (pd.DataFrame): A DataFrame containing the customer data for a specific route.
        vehicle_name (str): The name of the vehicle (e.g., 'small', 'medium').
        vehicles_df (pd.DataFrame): The DataFrame containing vehicle information.

    Returns:
        bool: True if the vehicle is compatible with all stores, False otherwise.
    """
    print(f"\n--- Checking Vehicle-Store Compatibility for Vehicle '{vehicle_name}' ---")
    
    # Get the PPL capacity of the current vehicle for comparison
    try:
        current_vehicle_ppl_total = vehicles_df.loc[vehicle_name, 'PPL total']
    except KeyError:
        print(f"Error: Vehicle name '{vehicle_name}' not found in the vehicle data.")
        return False

    # Define allowed PPL total values for 'Biltype' 2
    # These correspond to 'small' (17.5), 'medium-small' (20.0), 'medium' (22.0)
    # We should get these values directly from the vehicles_df for robustness
    allowed_ppl_for_biltype_2 = []
    for allowed_vehicle_name in ['small', 'medium-small', 'medium']:
        if allowed_vehicle_name in vehicles_df.index:
            allowed_ppl_for_biltype_2.append(vehicles_df.loc[allowed_vehicle_name, 'PPL total'])
    
    if not allowed_ppl_for_biltype_2:
        print("Error: Could not determine allowed PPL values for 'Biltype' 2 from vehicles_df.")
        return False


    for index, customer in route_customers_df.iterrows():
        # 'Biltype' 1 means compatible with all vehicles
        if customer['Biltype'] == 1:
            print(f"Store '{customer['Kundenavn']}' is compatible with all vehicles.")
            continue

        # 'Biltype' 2 means restricted compatibility: only small, medium-small, medium
        if customer['Biltype'] == 2:
            print(f"Store '{customer['Kundenavn']}' has restrictions. Allowed PPL for Biltype 2: {allowed_ppl_for_biltype_2}. Current Vehicle PPL: {current_vehicle_ppl_total}")
            if current_vehicle_ppl_total not in allowed_ppl_for_biltype_2:
                print(f"Result: Infeasible - Vehicle '{vehicle_name}' (PPL {current_vehicle_ppl_total}) is not compatible with store '{customer['Kundenavn']}' (Biltype 2).")
                return False
        else:
            # Handle other potential values for 'Biltype' if necessary
            # For now, if it's not 1 or 2, we assume incompatibility.
            print(f"Warning: Unhandled 'Biltype' value '{customer['Biltype']}' for store '{customer['Kundenavn']}'. Assuming incompatibility.")
            return False

    print("Result: Feasible")
    return True


if __name__ == "__main__":
    customers_df, vehicles_df, time_matrix_df, time_matrix_array, address_to_idx, customer_addr_idx, customer_arrays = load_vrp_data()
    
    if customers_df is not None and vehicles_df is not None and time_matrix_df is not None:
        print("\n--- Vehicle Capacity Information ---")

        # --- Example Demonstration ---
        # Create a simple example route with the first 5 customers
        # Use first 2 customers by index
        example_route_indices_small = [0, 1]
        check_capacity_feasibility(example_route_indices_small, vehicle_name='small', vehicles_df=vehicles_df, customer_arrays=customer_arrays)
        
        example_route_indices_large = [0, 1]
        check_capacity_feasibility(example_route_indices_large, vehicle_name='large', vehicles_df=vehicles_df, customer_arrays=customer_arrays)

        # --- Example Time Window Feasibility Check ---
        # Use the indices of the first 5 customers for the route
        example_route_indices = [0, 1, 2, 3, 4] 
        check_time_window_feasibility(
            example_route_indices,
            customers_df,
            time_matrix_array,
            address_to_idx,
            customer_addr_idx,
            customer_arrays,
            debug=True,
        )

