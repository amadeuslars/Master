from feasibilitycheck import load_vrp_data

def calculate_route_cost(solution, customer_addr_idx, time_matrix_array, depot_idx):
   
    total_travel_time = 0.0
    last_idx = depot_idx

    # Traverse route from depot through all customers
    for cust_row_idx in solution:
        current_idx = customer_addr_idx[cust_row_idx]
        total_travel_time += time_matrix_array[last_idx, current_idx]
        last_idx = current_idx

    # Return to depot if route is non-empty
    if solution:
        total_travel_time += time_matrix_array[last_idx, depot_idx]

    return total_travel_time


customers, vehicles, time_matrix_df, time_matrix_array, address_to_idx, customer_addr_idx, customer_arrays = load_vrp_data()
depot_idx = address_to_idx['Depot']
calculate_route_cost([1,2,3], customer_addr_idx, time_matrix_array, depot_idx)

