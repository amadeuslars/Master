"""
Simple ALNS for Solomon benchmark.
- Random removal
- Random insertion  
- Dummy vehicle as last route for unassigned customers
"""

import pandas as pd
import numpy as np
import random
import copy
from feasibility import (
    load_vrp_data,
    check_capacity_feasibility,
    check_time_window_feasibility
)
from cost import calculate_route_cost

# Configuration
DUMMY_VEHICLE_NAME = 'dummy'
DUMMY_PENALTY = 10000.0
MAX_ITERATIONS = 1000
PCT_REMOVE = 0.5  # Remove 10% of customers


class Solution:
    """
    Simple solution for Solomon instances.
    Structure: routes[0:24] = Real vehicles, routes[25] = Dummy vehicle
    """
    
    def __init__(self, routes, vehicles):
        self.routes = routes  # List of routes
        self.vehicles = vehicles  # Corresponding vehicle names
        self._cost = None
    
    def copy(self):
        """Shallow copy of solution."""
        return Solution(
            [r[:] for r in self.routes],
            self.vehicles[:]
        )
    
    @property
    def cost(self):
        """Get cached cost."""
        return self._cost
    
    def set_cost(self, cost):
        """Set cost."""
        self._cost = cost
    
    def get_unassigned(self):
        """Get customers in dummy route (last route)."""
        if self.routes and self.vehicles[-1] == DUMMY_VEHICLE_NAME:
            return self.routes[-1]
        return []


def evaluate_solution(solution, distance_matrix_array, customer_addr_idx, 
                     customer_arrays, depot_idx=0):
    """
    Evaluate solution cost.
    - Cost = sum of route distances  
    - Penalty = DUMMY_PENALTY * number of unassigned customers
    """
    total_cost = 0.0
    
    # Calculate cost for all routes
    for i, route in enumerate(solution.routes):
        if route and solution.vehicles[i] != DUMMY_VEHICLE_NAME: 
            route_cost = calculate_route_cost(
                route,
                customer_addr_idx,
                distance_matrix_array,
                depot_idx
            )
            total_cost += route_cost
    
    # Add penalty for unassigned customers (in dummy vehicle)
    unassigned = solution.get_unassigned()
    if unassigned:
        total_cost += len(unassigned) * DUMMY_PENALTY
    
    solution.set_cost(total_cost)
    return total_cost


def create_initial_solution(num_customers, num_real_vehicles):
    """
    Create initial solution with all customers in dummy vehicle.
    Structure: [empty_route1, ..., empty_route_n, dummy_route_with_all_customers]
    
    Note: Uses 1-indexed customers (1-100) to match Solomon benchmark format.
    """
    routes = [[] for _ in range(num_real_vehicles)]
    routes.append(list(range(1, num_customers + 1)))  # Customers 1-100 in dummy (last route)
    
    vehicles = ['Standard'] * num_real_vehicles + [DUMMY_VEHICLE_NAME]
    
    return Solution(routes, vehicles)


def random_removal(solution, num_to_remove):
    """
    Randomly remove customers from ANY route (including dummy).
    """
    new_sol = solution.copy()
    
    # Get all customers from all routes
    all_customers = []
    for route in new_sol.routes:
        all_customers.extend(route)
    
    if not all_customers:
        return new_sol
    
    # Randomly select customers to remove
    num_to_remove = min(num_to_remove, len(all_customers))
    customers_to_remove = set(random.sample(all_customers, num_to_remove))
    
    # Remove from all routes
    for i in range(len(new_sol.routes)):
        new_sol.routes[i] = [c for c in new_sol.routes[i] if c not in customers_to_remove]
    
    # Put removed customers in dummy (last route)
    new_sol.routes[-1].extend(customers_to_remove)
    
    return new_sol


def random_insertion(solution, distance_matrix_array, customer_addr_idx,
                    customer_arrays, depot_idx=0):
    """
    Greedily reinsert customers from dummy vehicle into ANY route.
    Try each customer in each route position (real + dummy), pick best feasible.
    """
    new_sol = solution.copy()
    
    # Get customers in dummy vehicle (last route)
    unassigned = list(new_sol.routes[-1])
    new_sol.routes[-1] = []  # Clear dummy
    
    random.shuffle(unassigned)
    
    # Try to insert each unassigned customer
    for customer_idx in unassigned:
        best_cost_increase = float('inf')
        best_move = None
        
        # Try inserting into ANY route (including dummy)
        for route_idx in range(len(new_sol.routes)):
            current_route = new_sol.routes[route_idx]
            
            # Try each position in route
            for pos in range(len(current_route) + 1):
                candidate_route = current_route[:pos] + [customer_idx] + current_route[pos:]
                
                # Check feasibility only for real routes, dummy always accepts
                is_feasible = True
                if route_idx < len(new_sol.routes) - 1:  # Not dummy route
                    is_feasible_time = check_time_window_feasibility(
                        candidate_route,
                        distance_matrix_array,
                        customer_addr_idx,
                        customer_arrays,
                        depot_idx
                    )
                    
                    if not is_feasible_time:
                        is_feasible = False
                    else:
                        is_feasible_capacity = check_capacity_feasibility(
                            candidate_route,
                            'Standard',
                            pd.DataFrame([{'capacity': 200}], index=['Standard']),
                            customer_arrays
                        )
                        
                        if not is_feasible_capacity:
                            is_feasible = False
                
                if not is_feasible:
                    continue
                
                # Calculate cost increase
                if route_idx < len(new_sol.routes) - 1:  # Real route
                    old_cost = calculate_route_cost(current_route, customer_addr_idx,
                                                   distance_matrix_array, depot_idx) if current_route else 0
                    new_cost = calculate_route_cost(candidate_route, customer_addr_idx,
                                                   distance_matrix_array, depot_idx)
                    cost_increase = new_cost - old_cost
                else:  # Dummy route (no distance cost, but penalty)
                    cost_increase = DUMMY_PENALTY if not current_route else 0
                
                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_move = (route_idx, pos)
        
        # Execute best move
        if best_move:
            route_idx, pos = best_move
            new_sol.routes[route_idx].insert(pos, customer_idx)
        else:
            # If no good move found, put in dummy
            new_sol.routes[-1].append(customer_idx)
    
    return new_sol


def run_simple_alns():
    """Run simple ALNS with random removal and insertion."""
    
    # Load data
    customers_df, vehicles_df, distance_matrix_df, distance_matrix_array, \
        customer_addr_idx, customer_arrays = load_vrp_data()
    
    num_customers = len(customers_df)
    num_real_vehicles = int(vehicles_df.loc['Standard', 'num_vehicles'])
    
    print(f"Customers: {num_customers}")
    print(f"Real vehicles: {num_real_vehicles}")
    print(f"Vehicle capacity: {vehicles_df.loc['Standard', 'capacity']}")
    
    # Create initial solution (all customers in dummy)
    print("\nCreating initial solution (all customers in dummy)...")
    current_sol = create_initial_solution(num_customers, num_real_vehicles)
    evaluate_solution(current_sol, distance_matrix_array, customer_addr_idx,
                     customer_arrays)
    
    print(f"Initial cost: {current_sol.cost:.2f}")
    print(f"Unassigned: {len(current_sol.get_unassigned())}")
    
    best_sol = current_sol.copy()
    best_sol.set_cost(current_sol.cost)
    
    # Main ALNS loop
    print(f"\nRunning {MAX_ITERATIONS} iterations...")
    
    for iteration in range(MAX_ITERATIONS):
        # Destroy
        num_remove = max(1, int(num_customers * PCT_REMOVE))
        destroyed_sol = random_removal(current_sol, num_remove)
        
        # Repair
        repaired_sol = random_insertion(destroyed_sol, distance_matrix_array,
                                       customer_addr_idx, customer_arrays)
        
        # Evaluate
        evaluate_solution(repaired_sol, distance_matrix_array, customer_addr_idx,
                         customer_arrays)
        
        # Accept if better
        if repaired_sol.cost < current_sol.cost:
            current_sol = repaired_sol
            
            if current_sol.cost < best_sol.cost:
                best_sol = current_sol.copy()
                best_sol.set_cost(current_sol.cost)
                unassigned = len(best_sol.get_unassigned())
                print(f"Iteration {iteration}: New best cost={best_sol.cost:.2f}, "
                      f"unassigned={unassigned}")
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Best cost: {best_sol.cost:.2f}")
    print(f"Unassigned customers: {len(best_sol.get_unassigned())}")
    print(f"Routes used: {len([r for r in best_sol.routes[:-1] if r])}")
    print(best_sol.routes)
    
    # Show route structure
    print("\nRoute structure:")
    for i, route in enumerate(best_sol.routes[:-1]):
        if route:
            cost = calculate_route_cost(route, customer_addr_idx,
                                       distance_matrix_array, 0)
            print(f"  Route {i+1} (Standard): {len(route)} customers, cost={cost:.2f}")
    
    unassigned = best_sol.get_unassigned()
    if unassigned:
        print(f"  Dummy vehicle: {len(unassigned)} unassigned customers, "
              f"cost={len(unassigned)*DUMMY_PENALTY:.2f}")


if __name__ == "__main__":
    run_simple_alns()