from .feasibility import check_capacity_feasibility, check_time_window_feasibility
from .cost import calculate_route_cost
import os
import pandas as pd
import numpy as np
from typing import Dict

class Solution:
    def __init__(self, routes, vehicles):
        self.routes = [r[:] for r in routes] 
        self.vehicles = vehicles
        self._cost = None
    
    def copy(self):
        return Solution(self.routes, self.vehicles)
    
class SolomonInstance:
    """Parse and store Solomon benchmark instance data."""
    
    def __init__(self, file_path: str):
        # Make path relative to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(script_dir, file_path)
        
        self.instance_name = None
        self.num_vehicles = None
        self.vehicle_capacity = None
        self.customers = []  # List of customer dictionaries
        self.depot = None
        
        self._parse_file()
    
    def _parse_file(self):
        """Parse Solomon instance file."""
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        
        # Get instance name
        self.instance_name = lines[0].strip()
        
        # Parse vehicle information
        for i, line in enumerate(lines):
            if 'VEHICLE' in line:
                capacity_line = lines[i + 2].strip().split()
                self.num_vehicles = int(capacity_line[0])
                self.vehicle_capacity = float(capacity_line[1])
                break
        
        # Parse customer data
        reading_customers = False
        
        for line in lines:
            if 'CUST NO.' in line:
                reading_customers = True
                continue
            
            if reading_customers and line.strip():
                parts = line.strip().split()
                if len(parts) >= 7:
                    customer = {
                        'id': int(parts[0]),
                        'x': float(parts[1]),
                        'y': float(parts[2]),
                        'demand': float(parts[3]),
                        'ready_time': float(parts[4]),
                        'due_date': float(parts[5]),
                        'service_time': float(parts[6])
                    }
                    
                    if customer['id'] == 0:
                        self.depot = customer
                    else:
                        self.customers.append(customer)
    
    def calculate_euclidean_distance(self, loc1: Dict, loc2: Dict) -> float:
        """Calculate Euclidean distance between two locations."""
        dx = loc1['x'] - loc2['x']
        dy = loc1['y'] - loc2['y']
        return np.sqrt(dx**2 + dy**2)
    
    def create_distance_matrix(self) -> pd.DataFrame:
        """
        Create distance matrix CSV similar to real-life data.
        Row 0/Column 0 is depot, then customers 1, 2, 3, ...
        """
        # All locations (depot + customers)
        all_locations = [self.depot] + self.customers
        n = len(all_locations)
        
        # Create distance matrix
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                distances[i, j] = self.calculate_euclidean_distance(
                    all_locations[i], 
                    all_locations[j]
                )
        
        # Create DataFrame with proper labels
        labels = ['Depot'] + [f'Customer_{c["id"]}' for c in self.customers]
        df = pd.DataFrame(distances, index=labels, columns=labels)
        
        return df
    
    def __repr__(self):
        return (f"SolomonInstance({self.instance_name}, "
                f"customers={len(self.customers)}, "
                f"vehicles={self.num_vehicles}, "
                f"capacity={self.vehicle_capacity})")
    

    
def evaluate_solution(solution, distance_matrix_array, customer_addr_idx, depot_idx=0):
    """Calculates total cost including penalties for dummy vehicle."""
    total_cost = 0.0
    
    # Real routes
    for i, route in enumerate(solution.routes[:-1]):
        if route:
            total_cost += calculate_route_cost(
                route, customer_addr_idx, distance_matrix_array, depot_idx
            )
    
    # Dummy route penalty
    unassigned_count = len(solution.routes[-1])
    total_cost += unassigned_count * 100
    
    solution._cost = total_cost
    return total_cost

def precompute_nearest_neighbors(distance_matrix_array, num_neighbors=20):
    n = distance_matrix_array.shape[0]
    k = min(num_neighbors, n - 1)
    neighbors = []
    for i in range(n):
        row = distance_matrix_array[i]
        idxs = np.argpartition(row, k)[:k+1]
        idxs = idxs[idxs != i]
        if idxs.shape[0] > k:
            order = np.argsort(row[idxs])[:k]
            idxs = idxs[order]
        else:
            order = np.argsort(row[idxs])
            idxs = idxs[order]
        neighbors.append(set(map(int, idxs.tolist())))
    return neighbors

def simple_relocate(solution, distance_matrix_array, customer_addr_idx, customer_arrays, vehicles_df, depot_idx=0):
    """
    INTER-ROUTE: Moves single customers between routes.
    Runs faster than segment relocation, good for continuous balancing.
    """
    improved = False
    # Handle both dict and DataFrame formats
    if isinstance(vehicles_df, dict):
        capacity = vehicles_df['capacity']
        capacities = [capacity] * (len(solution.routes) - 1)
    else:
        capacities = [vehicles_df.loc[v, 'capacity'] for v in solution.vehicles[:-1]]
    
    for r_src_idx, src_route in enumerate(solution.routes[:-1]):
        if not src_route: continue
        
        for i, cust in enumerate(src_route):
            cust_demand = customer_arrays['demand'][cust-1]
            cust_addr = customer_addr_idx[cust-1]
            
            # Calculate savings of removing customer
            cost_pre = calculate_route_cost(src_route, customer_addr_idx, distance_matrix_array, depot_idx)
            temp_src = src_route[:i] + src_route[i+1:]
            cost_post = calculate_route_cost(temp_src, customer_addr_idx, distance_matrix_array, depot_idx)
            savings = cost_pre - cost_post
            
            # Try inserting into other routes
            for r_dst_idx, dst_route in enumerate(solution.routes[:-1]):
                if r_src_idx == r_dst_idx: continue
                
                # 1. Capacity Check
                dst_demand = sum(customer_arrays['demand'][c-1] for c in dst_route)
                if dst_demand + cust_demand > capacities[r_dst_idx]: continue
                
                # 2. Find best insertion
                best_delta = float('inf')
                best_pos = -1
                
                dst_addrs = [depot_idx] + [customer_addr_idx[c-1] for c in dst_route] + [depot_idx]
                
                for j in range(len(dst_route) + 1):
                    prev = dst_addrs[j]
                    next_node = dst_addrs[j+1]
                    
                    # Marginal cost of insertion
                    increase = (distance_matrix_array[prev, cust_addr] + 
                                distance_matrix_array[cust_addr, next_node] - 
                                distance_matrix_array[prev, next_node])
                    
                    # Net change
                    delta = increase - savings
                    
                    if delta < -1e-3: # Improvement found
                        if delta < best_delta:
                            # 3. Feasibility Check (Lazy)
                            candidate_dst = dst_route[:j] + [cust] + dst_route[j:]
                            if check_time_window_feasibility(candidate_dst, distance_matrix_array, customer_addr_idx, customer_arrays, depot_idx):
                                best_delta = delta
                                best_pos = j
                
                if best_pos != -1:
                    solution.routes[r_src_idx] = temp_src
                    solution.routes[r_dst_idx].insert(best_pos, cust)
                    return True # First improvement
                    
    return False

def cross_route_segment_relocation(solution, distance_matrix_array, customer_addr_idx, customer_arrays, vehicles_df, depot_idx=0):
    """
    HEAVY INTER-ROUTE: Relocates segments (chains of 1-3 customers).
    Computational expensive, run periodically.
    """
    improved = False
    # Handle both dict and DataFrame formats
    if isinstance(vehicles_df, dict):
        capacity = vehicles_df['capacity']
        capacities = [capacity] * (len(solution.routes) - 1)
    else:
        capacities = [vehicles_df.loc[v, 'capacity'] for v in solution.vehicles[:-1]]
    
    # Try relocating segments of 1, 2, 3 customers
    for seg_len in [1, 2, 3]:
        for r_idx in range(len(solution.routes) - 1):
            route_src = solution.routes[r_idx]
            if len(route_src) < seg_len: continue
            
            for seg_start in range(len(route_src) - seg_len + 1):
                segment = route_src[seg_start:seg_start + seg_len]
                
                # Source route without segment
                new_src = route_src[:seg_start] + route_src[seg_start + seg_len:]
                src_cost_old = calculate_route_cost(route_src, customer_addr_idx, distance_matrix_array, depot_idx)
                
                for r_dst in range(len(solution.routes) - 1):
                    if r_dst == r_idx: continue
                    route_dst = solution.routes[r_dst]
                    
                    for pos in range(len(route_dst) + 1):
                        new_dst = route_dst[:pos] + segment + route_dst[pos:]
                        
                        # Capacity Check
                        src_demand = sum(customer_arrays['demand'][c-1] for c in new_src)
                        dst_demand = sum(customer_arrays['demand'][c-1] for c in new_dst)
                        if src_demand > capacities[r_idx] or dst_demand > capacities[r_dst]:
                            continue
                        
                        # Time Window Check
                        if not check_time_window_feasibility(new_src, distance_matrix_array, customer_addr_idx, customer_arrays, depot_idx): continue
                        if not check_time_window_feasibility(new_dst, distance_matrix_array, customer_addr_idx, customer_arrays, depot_idx): continue
                        
                        # Cost Check
                        src_cost_new = calculate_route_cost(new_src, customer_addr_idx, distance_matrix_array, depot_idx)
                        dst_cost_old = calculate_route_cost(route_dst, customer_addr_idx, distance_matrix_array, depot_idx)
                        dst_cost_new = calculate_route_cost(new_dst, customer_addr_idx, distance_matrix_array, depot_idx)
                        
                        delta = (src_cost_new - src_cost_old) + (dst_cost_new - dst_cost_old)
                        
                        if delta < -1e-3:
                            solution.routes[r_idx] = new_src
                            solution.routes[r_dst] = new_dst
                            return True
    return improved

def two_opt_local_search(solution, distance_matrix_array, customer_addr_idx, customer_arrays, depot_idx=0):
    """INTRA-ROUTE: Untangles crossings within single routes."""
    improved = False
    for r_idx, route in enumerate(solution.routes[:-1]):
        if len(route) < 3: continue
        
        best_route_cost = calculate_route_cost(route, customer_addr_idx, distance_matrix_array, depot_idx)
        route_improved = True
        
        while route_improved:
            route_improved = False
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    if j - i == 1: continue 
                    
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_cost = calculate_route_cost(new_route, customer_addr_idx, distance_matrix_array, depot_idx)
                    
                    if new_cost < best_route_cost - 1e-4:
                        if check_time_window_feasibility(new_route, distance_matrix_array, customer_addr_idx, customer_arrays, depot_idx):
                            route[:] = new_route 
                            best_route_cost = new_cost
                            route_improved = True
                            improved = True
                            break 
                if route_improved: break
    return improved

def create_initial_solution(num_customers, num_real_vehicles):
    routes = [[] for _ in range(num_real_vehicles)]
    routes.append(list(range(1, num_customers + 1)))
    vehicles = ['Standard'] * num_real_vehicles + ['dummy']
    return Solution(routes, vehicles)

def load_raw_solomon_data(file_path):
    """
    Parses a raw Solomon .txt file and returns the data structures 
    required by the VRPTWEnv. Skips CSV conversion and goes directly 
    from parsed .txt to numpy arrays.
    """
    # 1. Parse the text file
    instance = SolomonInstance(file_path)
    
    # 2. Create Vehicles dict (minimal structure)
    vehicles_df = {
        'vehicle_type': 'Standard',
        'capacity': instance.vehicle_capacity,
        'num_vehicles': instance.num_vehicles
    }
    
    # 3. Build numpy arrays directly from parsed customers
    n_customers = len(instance.customers)
    customer_ids = np.array([c['id'] for c in instance.customers], dtype=np.int32)
    customer_x = np.array([c['x'] for c in instance.customers], dtype=np.float32)
    customer_y = np.array([c['y'] for c in instance.customers], dtype=np.float32)
    
    # 4. Build distance matrix directly (Depot is index 0, customers 1..n)
    all_locs = [(instance.depot['x'], instance.depot['y'])] + [(c['x'], c['y']) for c in instance.customers]
    n = len(all_locs)
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        for j in range(n):
            dx = all_locs[i][0] - all_locs[j][0]
            dy = all_locs[i][1] - all_locs[j][1]
            dist_matrix[i, j] = np.sqrt(dx**2 + dy**2)
    
    # 5. Customer address indices (customer ID to matrix index mapping)
    customer_addr_idx = customer_ids.astype(np.int32)
    
    # 6. Build customer arrays dict
    cust_arrays = {
        'demand': np.array([c['demand'] for c in instance.customers], dtype=np.float32),
        'tw_start': np.array([c['ready_time'] for c in instance.customers], dtype=np.float32),
        'tw_end': np.array([c['due_date'] for c in instance.customers], dtype=np.float32),
        'service_time': np.array([c['service_time'] for c in instance.customers], dtype=np.float32),
    }
    
    # 7. Return minimal customers_df for compatibility
    customers_df = {
        'customer_id': customer_ids,
        'x_coord': customer_x,
        'y_coord': customer_y,
    }
    
    return customers_df, vehicles_df, dist_matrix, customer_addr_idx, cust_arrays