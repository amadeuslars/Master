# cost.py
# (No imports needed here if we just do math, but keeping structure)

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