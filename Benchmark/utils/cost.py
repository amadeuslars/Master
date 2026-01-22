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
    
# customers_df, vehicles_df, distance_matrix_df, distance_matrix_array, customer_addr_idx, customer_arrays = load_vrp_data()

# RC105 Best Known Solution (Bent & Van Hentenryck, 2001)
# 13 vehicles, distance: 1513.7
# solution = [
#     [90, 53, 66, 56],
#     [63, 62, 67, 84, 51, 85, 91],
#     [72, 71, 81, 41, 54, 96, 94, 93],
#     [65, 82, 12, 11, 87, 59, 97, 75, 58],
#     [33, 76, 89, 48, 21, 25, 24],
#     [98, 14, 47, 15, 16, 9, 10, 13, 17],
#     [42, 61, 8, 6, 46, 4, 3, 1, 100],
#     [39, 36, 44, 38, 40, 37, 35, 43, 70],
#     [83, 19, 23, 18, 22, 49, 20, 77],
#     [31, 29, 27, 30, 28, 26, 32, 34, 50, 80],
#     [92, 95, 64, 99, 52, 86, 57, 74],
#     [69, 88, 78, 73, 60],
#     [2, 45, 5, 7, 79, 55, 68]
# ]

# total_cost = 0.0
# for i, route in enumerate(solution, 1):
#     route_cost = calculate_route_cost(
#         route,
#         customer_addr_idx,
#         distance_matrix_array,
#         depot_idx=0
#     )
#     total_cost += route_cost
#     print(f"Route {i}: {len(route)} customers, cost={route_cost:.2f}")

# print(f"\nTotal distance: {total_cost:.2f}")
# print(f"Number of vehicles: {len(solution)}")