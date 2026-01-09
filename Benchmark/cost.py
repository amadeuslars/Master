from feasibility import load_vrp_data

load_vrp_data()

def calculate_route_cost(solution, customer_addr_idx, time_matrix_array, depot_idx):
    """
    Calculates the total travel time for a route.
    Optimized for raw list/array access.
    """
    if not solution:
        return 0.0

    total_travel_time = 0.0
    last_idx = depot_idx

    # Traverse route from depot through all customers
    # solution is expected to be a list of customer INDICES (ints)
    for cust_row_idx in solution:
        current_idx = customer_addr_idx[cust_row_idx]
        total_travel_time += time_matrix_array[last_idx, current_idx]
        last_idx = current_idx

    # Return to depot
    total_travel_time += time_matrix_array[last_idx, depot_idx]

    return total_travel_time
    
customers, vehicles, distance_matrix, distance_matrix_array, customer_addr_idx, customer_arrays = load_vrp_data()

print(calculate_route_cost([2, 1, 0], customer_addr_idx, distance_matrix_array, 0))