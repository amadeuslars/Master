import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import Solution, load_vrp_data, evaluate_solution

# Load data to get vehicle names and other info
customers_dict, vehicles_dict, vehicle_names, time_matrix_array, distance_matrix_array, depot_idx, addr_idx, customer_arrays = load_vrp_data()

# Example: assign first N vehicles to routes, fill with 'dummy' if needed
num_routes = 11
vehicles = vehicle_names[:num_routes] + ['dummy'] * max(0, num_routes - len(vehicle_names))

routes = [
    [1, 2],
    [46, 4, 5, 47, 3, 41, 7, 42, 43, 44, 45, 6],
    [10, 9, 8],
    [48, 11, 13, 12, 14, 49],
    [50, 51, 52, 53, 18, 54, 15, 16, 56, 17, 57, 58, 59, 19, 60, 61],
    [21, 20],
    [22, 23, 62],
    [24],
    [25, 26],
    [30, 31, 28, 63, 27, 32, 64, 29, 33, 34, 35],
    [38, 37, 65, 66, 67, 68, 69, 70, 71, 72, 40, 39]
]

current = Solution(routes, vehicles)
evaluate_solution(current, addr_idx, time_matrix_array, depot_idx)
print(f"Cost: {current.cost}")