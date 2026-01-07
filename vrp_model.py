import pandas as pd

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

    return customers_df, vehicles_df, time_matrix_df
    
def check_capacity_feasibility(route_customers_df, vehicle_name, vehicles_df):
    """
    Checks if the total demand of a route exceeds the capacity of the assigned vehicle.

    Args:
        route_customers_df (pd.DataFrame): A DataFrame containing the customer data for a specific route.
        vehicle_name (str): The name of the vehicle (e.g., 'small', 'medium').
        vehicles_df (pd.DataFrame): The DataFrame containing vehicle capacity information.

    Returns:
        bool: True if the route is feasible in terms of capacity, False otherwise.
    """
    if route_customers_df.empty:
        return True # An empty route is always feasible

    try:
        # Get the capacity constraints for the given vehicle name
        vehicle_capacities = vehicles_df.loc[vehicle_name]
        
        # Sum the demands for the customers on the route
        total_pallets = route_customers_df['PPL pr leveranse'].sum()
        total_volume = route_customers_df['Volum pr leveranse (m3)'].sum()
        total_weight_kg = route_customers_df['Vekt pr leveranse (tonn)'].sum() * 1000
        total_frys = route_customers_df['PPL hvorav frys'].sum()

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

def check_time_window_feasibility(route_customer_indices, customers_df, time_matrix_df, service_time=0.5):
    """
    Checks if a route is feasible regarding customer time windows.

    Args:
        route_customer_indices (list): A list of customer indices defining the order of the route.
        customers_df (pd.DataFrame): DataFrame with all customer data, including time windows.
        time_matrix_df (pd.DataFrame): DataFrame with travel times between all locations.
        service_time (float): The time spent at each customer location (in hours).

    Returns:
        bool: True if the route is feasible, False otherwise.
    """
    if not route_customer_indices:
        return True

    first_idx = route_customer_indices[0]
    first_cust_data = customers_df.iloc[first_idx]
    
    try:
        # Get travel time from Depot to First Customer (in hours)
        travel_to_first = time_matrix_df.loc['Depot', first_cust_data['Adresse']]
    except KeyError:
        # print("Error: Missing travel time data for Depot -> First Customer")
        return False

    # Logic: If Customer opens at 8.0 and travel is 1.0, we depart at 7.0.
    # We use max(0.0, ...) because we can't start at negative time.
    start_time = max(0.0, first_cust_data['tw_start_hours'] - travel_to_first)
    
    current_time = start_time
    last_location_name = 'Depot'

    print(f"\n--- Checking Time Window Feasibility ---")


    for customer_index in route_customer_indices:
        customer_data = customers_df.iloc[customer_index]
        customer_address = customer_data['Adresse']

        # 1. Calculate travel time from the last location to the current customer
        travel_time_hours = time_matrix_df.loc[last_location_name, customer_address] 
            
        # 2. Calculate arrival time at the customer
        arrival_time = current_time + travel_time_hours
        
        # 3. Get customer's time window and convert to hours
        tw_start = customer_data['tw_start_hours']
        tw_end = customer_data['tw_end_hours']

        print(f"Visiting '{customer_address}':")
        print(f"  Travel from '{last_location_name}' took {travel_time_hours:.2f}h.")
        print(f"  Arrival Time: {arrival_time:.2f}, Time Window: [{tw_start:.2f} - {tw_end:.2f}]")

        # 4. Check for time window violation
        # If we arrive before the time window opens, we wait.
        if arrival_time < tw_start:
            print(f"  Arrived early. Waiting until {tw_start:.2f}.")
            current_time = tw_start
        # If we arrive after the time window closes, the route is infeasible.
        elif arrival_time > tw_end:
            print(f"  Result: Infeasible - Arrival time {arrival_time:.2f} is after time window closes at {tw_end:.2f}.")
            return False
        # If we arrive within the time window.
        else:
            current_time = arrival_time

        # 5. Add service time to the current time to get departure time
        current_time += service_time
        print(f"  Departure Time: {current_time:.2f}")

        # 6. Update the last location for the next iteration
        last_location_name = customer_address
        
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

def calculate_route_cost(route_customer_indices, customers_df, time_matrix_df):
    """
    Calculates the total cost of a route, defined as the total travel time.

    Args:
        route_customer_indices (list): A list of customer indices defining the order of the route.
        customers_df (pd.DataFrame): DataFrame with all customer data.
        time_matrix_df (pd.DataFrame): DataFrame with travel times between all locations.

    Returns:
        float: The total travel time for the route in hours.
    """
    total_travel_time = 0.0
    last_location_name = 'Depot' # Assuming the depot is named 'Depot' and is the last row/column in the matrix

    # Travel from depot to the first customer
    if route_customer_indices:
        first_customer_address = customers_df.iloc[route_customer_indices[0]]['Adresse']
        total_travel_time += time_matrix_df.loc[last_location_name, first_customer_address]
        last_location_name = first_customer_address

    # Travel between customers
    for i in range(len(route_customer_indices) - 1):
        current_customer_address = customers_df.iloc[route_customer_indices[i]]['Adresse']
        next_customer_address = customers_df.iloc[route_customer_indices[i+1]]['Adresse']
        total_travel_time += time_matrix_df.loc[current_customer_address, next_customer_address]

    # Travel from the last customer back to the depot
    if route_customer_indices:
        last_customer_address = customers_df.iloc[route_customer_indices[-1]]['Adresse']
        depot_name = 'Depot' # Assuming the depot is named 'Depot'
        total_travel_time += time_matrix_df.loc[last_customer_address, depot_name] 

    return total_travel_time


if __name__ == "__main__":
    customers_df, vehicles_df, time_matrix_df = load_vrp_data()
    
    if customers_df is not None and vehicles_df is not None and time_matrix_df is not None:
        print("\n--- Vehicle Capacity Information ---")

        # --- Example Demonstration ---
        # Create a simple example route with the first 5 customers
        example_route_df = customers_df.head(2)
        
        # Check feasibility for a 'small' vehicle
        check_capacity_feasibility(example_route_df, vehicle_name='small', vehicles_df=vehicles_df)
        
        # Check feasibility for a 'large' vehicle
        check_capacity_feasibility(example_route_df, vehicle_name='large', vehicles_df=vehicles_df)

        # --- Example Time Window Feasibility Check ---
        # Use the indices of the first 5 customers for the route
        example_route_indices = [0, 1, 2, 3, 4] 
        check_time_window_feasibility(example_route_indices, customers_df, time_matrix_df)

        # --- Example Route Cost Calculation ---
        print("\n--- Testing Route Cost Calculation ---")
        route_cost = calculate_route_cost(example_route_indices, customers_df, time_matrix_df)
        print(f"Example Route Cost (total travel time): {route_cost:.2f} hours")
