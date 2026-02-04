# from feasibility import load_vrp_data

def calculate_route_cost(solution, customer_addr_idx, time_matrix_array, depot_idx):
    """
    Calculates the total travel time for a route.
    Optimized for raw list/array access.
    
    Note: solution contains 1-indexed customer indices (1-100 for 100 customers, Solomon format)
    """
    if not solution:
        return 0.0

    total_travel_time = 0.0
    last_idx = depot_idx

    # Traverse route from depot through all customers
    # solution is expected to be a list of 1-indexed customer IDs (1-100 for 100 customers)
    for cust_idx in solution:
        current_idx = customer_addr_idx[cust_idx-1]
        total_travel_time += time_matrix_array[last_idx, current_idx]
        last_idx = current_idx

    # Return to depot
    total_travel_time += time_matrix_array[last_idx, depot_idx]

    return total_travel_time
    
