"""
Check a single route's total drive time.

Edit ROUTE below with customer numbers (Kundenr) in visit order.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import load_vrp_data
from utils.cost import calculate_route_cost
from utils.feasibility import (
    check_time_window_feasibility,
    check_max_route_duration,
    find_lunch_break_position,
)

# ── Edit this ────────────────────────────────────────────────────────────────
ROUTE = [8280, 8942, 8343, 8378, 9238, 9178, 4225, 9295, 9045]    # Kundenr values in visit order
# ─────────────────────────────────────────────────────────────────────────────

customers_dict, _, _, time_matrix_array, _, depot_idx, addr_idx, customer_arrays = load_vrp_data(
    customers_file='Case_study/data/customers.csv',
    vehicles_file='Case_study/data/vehicles.csv',
    time_matrix_file='Case_study/data/time_matrix.csv',
    dist_matrix_file='Case_study/data/distance_matrix.csv',
)

# Map Kundenr -> 1-based row index
kundenr_to_idx = {int(k): i + 1 for i, k in enumerate(customers_dict['customer_id'])}

route_indices = []
for knr in ROUTE:
    if knr not in kundenr_to_idx:
        raise ValueError(f"Customer number {knr} not found in data")
    route_indices.append(kundenr_to_idx[knr])

total_hours = calculate_route_cost(route_indices, addr_idx, time_matrix_array, depot_idx)
total_minutes = total_hours * 60

tw_ok      = check_time_window_feasibility(route_indices, time_matrix_array, addr_idx, customer_arrays, depot_idx)
dur_ok     = check_max_route_duration(route_indices, addr_idx, time_matrix_array, depot_idx)
lunch_pos  = find_lunch_break_position(route_indices, time_matrix_array, addr_idx, customer_arrays, depot_idx)
lunch_ok   = lunch_pos is not None

print(f"Route (Kundenr):  {ROUTE}")
print(f"Drive time:       {total_hours:.2f} h  ({total_minutes:.1f} min)")
print(f"Time windows:     {'OK' if tw_ok  else 'INFEASIBLE'}")
print(f"Max duration:     {'OK' if dur_ok else 'INFEASIBLE'}")
print(f"Lunch break:      {'OK (after position ' + str(lunch_pos) + ')' if lunch_ok else 'INFEASIBLE'}")
print(f"Overall:          {'FEASIBLE' if tw_ok and dur_ok and lunch_ok else 'INFEASIBLE'}")
