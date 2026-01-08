import pandas as pd
import numpy as np
import random
import copy
import sys
import os
from collections import Counter, defaultdict

# --- Configuration ---
RANDOM_SEED = 42
MAX_ITERATIONS = 100  
PCT_REMOVE = 0.3      
DUMMY_VEHICLE_NAME = 'dummy_virtual'
DUMMY_PENALTY = 10000.0  

# --- Imports ---
try:
    from feasibilitycheck import (
        load_vrp_data, 
        check_capacity_feasibility, 
        check_time_window_feasibility, 
        check_vehicle_store_compatibility 
    )
    from cost import calculate_route_cost
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

random.seed(RANDOM_SEED)

class Solution:
    def __init__(self, routes, vehicles, unassigned=None):
        self.routes = routes  # List of lists (customer indices)
        self.vehicles = vehicles  # List of vehicle names
        self.unassigned = unassigned if unassigned is not None else []
        self.cost = 0.0
    
    def get_vehicle_counts(self):
        return Counter(self.vehicles)

def suppress_output(func, *args, **kwargs):
    """Helper to run verbose functions silently."""
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout
    return result

# --- Core Logic ---

def check_compatibility_wrapper(route_data, vehicle_name, vehicles_df):
    if vehicle_name == DUMMY_VEHICLE_NAME:
        return True
    return suppress_output(check_vehicle_store_compatibility, route_data, vehicle_name, vehicles_df)

def evaluate_solution(solution, customer_addr_idx, time_matrix_array, depot_idx):
    total_cost = 0.0
    for i, route in enumerate(solution.routes):
        route_cost = calculate_route_cost(route, customer_addr_idx, time_matrix_array, depot_idx)
        total_cost += route_cost
        if solution.vehicles[i] == DUMMY_VEHICLE_NAME:
            total_cost += DUMMY_PENALTY
    solution.cost = total_cost
    return total_cost

def get_best_available_vehicle(route_indices, customers_df, vehicles_df, current_usage, customer_arrays, vehicle_to_release=None):
    if not route_indices:
        return None

    route_data = customers_df.iloc[route_indices]
    
    # Sort: Real vehicles (by size) -> then Dummy
    real_vehicles = vehicles_df[vehicles_df.index != DUMMY_VEHICLE_NAME].sort_values(by='PPL total')
    sorted_vehicle_names = list(real_vehicles.index) + [DUMMY_VEHICLE_NAME]
    
    for vehicle_name in sorted_vehicle_names:
        limit = vehicles_df.loc[vehicle_name, 'antall']
        used = current_usage[vehicle_name]
        
        if vehicle_name == vehicle_to_release:
            used -= 1 
            
        if used >= limit:
            continue 

        if not check_compatibility_wrapper(route_data, vehicle_name, vehicles_df):
            continue
        
        valid_capacity = suppress_output(
            check_capacity_feasibility, 
            route_indices, 
            vehicle_name, 
            vehicles_df, 
            customer_arrays
        )
        if valid_capacity:
            return vehicle_name
            
    return None

# --- Operators ---

def random_removal(solution, num_to_remove):
    new_sol = copy.deepcopy(solution)
    all_routed = [c for r in new_sol.routes for c in r]
    
    if not all_routed: return new_sol

    num_to_remove = min(num_to_remove, len(all_routed))
    removed_customers = random.sample(all_routed, num_to_remove)
    new_sol.unassigned.extend(removed_customers)
    
    updated_routes = []
    updated_vehicles = []
    
    for r_idx, route in enumerate(new_sol.routes):
        new_route = [c for c in route if c not in removed_customers]
        if new_route:
            updated_routes.append(new_route)
            updated_vehicles.append(new_sol.vehicles[r_idx])
            
    new_sol.routes = updated_routes
    new_sol.vehicles = updated_vehicles
    return new_sol

def greedy_insertion(solution, customers_df, vehicles_df, time_matrix_df, 
                     customer_addr_idx, time_matrix_array, depot_idx, 
                     address_to_idx, customer_arrays):
    new_sol = copy.deepcopy(solution)
    random.shuffle(new_sol.unassigned)
    
    current_usage = new_sol.get_vehicle_counts()
    unassigned_list = list(new_sol.unassigned)
    new_sol.unassigned = []
    
    for customer_idx in unassigned_list:
        best_cost_increase = float('inf')
        best_move = None 
        
        # 1. Try inserting into existing routes
        for r_idx, route in enumerate(new_sol.routes):
            current_vehicle = new_sol.vehicles[r_idx]
            
            for i in range(len(route) + 1):
                temp_route = route[:i] + [customer_idx] + route[i:]
                
                is_feasible_time = suppress_output(
                    check_time_window_feasibility,
                    temp_route,
                    customers_df,
                    time_matrix_array,
                    address_to_idx,
                    customer_addr_idx,
                    customer_arrays
                )
                
                if not is_feasible_time:
                    continue
                
                needed_vehicle = get_best_available_vehicle(
                    temp_route, customers_df, vehicles_df, current_usage, customer_arrays, vehicle_to_release=current_vehicle
                )
                
                if not needed_vehicle:
                    continue
                
                old_cost = calculate_route_cost(route, customer_addr_idx, time_matrix_array, depot_idx)
                new_cost = calculate_route_cost(temp_route, customer_addr_idx, time_matrix_array, depot_idx)
                base_cost_diff = new_cost - old_cost
                
                penalty_diff = 0
                if needed_vehicle == DUMMY_VEHICLE_NAME and current_vehicle != DUMMY_VEHICLE_NAME:
                    penalty_diff += DUMMY_PENALTY
                elif needed_vehicle != DUMMY_VEHICLE_NAME and current_vehicle == DUMMY_VEHICLE_NAME:
                    penalty_diff -= DUMMY_PENALTY
                
                total_increase = base_cost_diff + penalty_diff
                
                if total_increase < best_cost_increase:
                    best_cost_increase = total_increase
                    best_move = ('existing', r_idx, i, needed_vehicle)

        # 2. Try creating a new route
        new_route = [customer_idx]
        needed_vehicle_new = get_best_available_vehicle(
            new_route, customers_df, vehicles_df, current_usage, customer_arrays, vehicle_to_release=None
        )
        
        if needed_vehicle_new:
            is_feasible_time_new = suppress_output(
                check_time_window_feasibility,
                new_route,
                customers_df,
                time_matrix_array,
                address_to_idx,
                customer_addr_idx,
                customer_arrays
            )
            
            if is_feasible_time_new:
                base_cost = calculate_route_cost(new_route, customer_addr_idx, time_matrix_array, depot_idx)
                penalty = DUMMY_PENALTY if needed_vehicle_new == DUMMY_VEHICLE_NAME else 0
                total_cost = base_cost + penalty
                
                if total_cost < best_cost_increase:
                    best_cost_increase = total_cost
                    best_move = ('new', -1, 0, needed_vehicle_new)

        # 3. Execute
        if best_move:
            m_type, r_idx, pos, vehicle = best_move
            if m_type == 'existing':
                new_sol.routes[r_idx].insert(pos, customer_idx)
                old_vehicle = new_sol.vehicles[r_idx]
                if old_vehicle != vehicle:
                    current_usage[old_vehicle] -= 1
                    current_usage[vehicle] += 1
                    new_sol.vehicles[r_idx] = vehicle
            else:
                new_sol.routes.append([customer_idx])
                new_sol.vehicles.append(vehicle)
                current_usage[vehicle] += 1
        else:
            new_sol.unassigned.append(customer_idx)

    return new_sol

# --- Main & Formatting ---

def generate_initial_dummy_solution(customers_df, customer_addr_idx, time_matrix_array, depot_idx):
    print("Generating initial DUMMY solution...")
    routes = []
    vehicles = []
    for i in range(len(customers_df)):
        routes.append([i])
        vehicles.append(DUMMY_VEHICLE_NAME)
    
    sol = Solution(routes, vehicles)
    evaluate_solution(sol, customer_addr_idx, time_matrix_array, depot_idx)
    return sol

def format_solution_flat(solution, customers_df, vehicles_df):
    """
    Formats the solution to have exactly 26 separators (0).
    Structure: 
      - Iterate through real vehicle types (Small, Medium-Small, etc.)
      - For each type, assume 5 available vehicles.
      - Fill slots with actual routes, leave empty if unused.
      - Finally, dump all Dummy Vehicle routes into the 26th slot (or multiple if needed).
    """
    flat_list = []
    
    # 1. Group routes by vehicle name
    routes_by_vehicle = defaultdict(list)
    dummy_routes = []
    
    for route, veh_name in zip(solution.routes, solution.vehicles):
        if veh_name == DUMMY_VEHICLE_NAME:
            dummy_routes.append(route)
        else:
            routes_by_vehicle[veh_name].append(route)
            
    # 2. Iterate through fixed vehicle types order (excluding dummy)
    # We use the order from the DataFrame to be deterministic
    real_vehicle_types = [v for v in vehicles_df.index if v != DUMMY_VEHICLE_NAME]
    
    for v_type in real_vehicle_types:
        # Get all routes assigned to this type
        assigned_routes = routes_by_vehicle.get(v_type, [])
        
        # We have 5 slots per vehicle type
        for i in range(5):
            if i < len(assigned_routes):
                # Add customers for this route
                route = assigned_routes[i]
                for customer_idx in route:
                    kundenr = customers_df.iloc[customer_idx]['Kundenr']
                    flat_list.append(int(kundenr))
            
            # Always add a separator '0' after every vehicle slot (used or unused)
            flat_list.append(0)

    # 3. Add Dummy Vehicle(s)
    # The requirement is "The last 0 is the dummy vehicle". 
    # We put ALL dummy customers into one block if multiple exist, or just empty if none.
    for route in dummy_routes:
        for customer_idx in route:
            kundenr = customers_df.iloc[customer_idx]['Kundenr']
            flat_list.append(int(kundenr))
            
    # Add the final 26th separator
    flat_list.append(0)
    
    return flat_list

def run_alns():
    print("--- ALNS VRP Optimizer (with Dummy Vehicle) ---")
    
    try:
        data = load_vrp_data()
        customers_df = data[0]
        vehicles_df = data[1]
        time_matrix_df = data[2]
        time_matrix_array = data[3]
        address_to_idx = data[4]
        customer_addr_idx = data[5]
        customer_arrays = data[6]
        
        depot_idx = address_to_idx.get('Depot', 0)
            
    except ValueError:
        print("Error: load_vrp_data did not return 7 values.")
        sys.exit(1)

    vehicles_df.index = vehicles_df.index.str.strip()
    if 'antall' not in vehicles_df.columns:
        vehicles_df['antall'] = 5

    vehicles_df.loc[DUMMY_VEHICLE_NAME] = {
        'PPL total': 9999.0, 'PPL Frys': 9999.0, 'm3': 9999.0, 'Vekt (KG)': 99999.0, 
        'antall': 9999
    }

    current_sol = generate_initial_dummy_solution(customers_df, customer_addr_idx, time_matrix_array, depot_idx)
    best_sol = copy.deepcopy(current_sol)
    
    print(f"Initial Cost: {current_sol.cost:.2f}")
    
    for i in range(MAX_ITERATIONS):
        n_remove = max(1, int(sum(len(r) for r in current_sol.routes) * PCT_REMOVE))
        destroyed_sol = random_removal(current_sol, n_remove)
        
        repaired_sol = greedy_insertion(
            destroyed_sol, 
            customers_df, vehicles_df, time_matrix_df, 
            customer_addr_idx, time_matrix_array, depot_idx,
            address_to_idx, customer_arrays
        )
        
        evaluate_solution(repaired_sol, customer_addr_idx, time_matrix_array, depot_idx)
        
        if repaired_sol.cost < current_sol.cost:
            current_sol = repaired_sol
            if current_sol.cost < best_sol.cost:
                best_sol = copy.deepcopy(current_sol)
                print(f"Iter {i}: New Best Cost={best_sol.cost:.2f}")

    print("\n--- Final Results ---")
    print(f"Best Cost: {best_sol.cost:.2f}")
    
    # Pass vehicles_df to format function to get correct order
    formatted_output = format_solution_flat(best_sol, customers_df, vehicles_df)
    
    print(f"\nFormatted Solution ({formatted_output.count(0)} separators):")
    print(formatted_output)

if __name__ == "__main__":
    run_alns()