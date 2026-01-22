from feasibility import check_capacity_feasibility, check_time_window_feasibility
from cost import calculate_route_cost

class Solution:
    def __init__(self, routes, vehicles):
        self.routes = [r[:] for r in routes] 
        self.vehicles = vehicles
        self._cost = None
    
    def copy(self):
        return Solution(self.routes, self.vehicles)
    
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
    total_cost += unassigned_count * DUMMY_PENALTY
    
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
    vehicles = ['Standard'] * num_real_vehicles + [DUMMY_VEHICLE_NAME]
    return Solution(routes, vehicles)

def load_vrp_data(customers_file='c1_8_1/customers.csv', vehicles_file='c1_8_1/vehicles.csv', distance_matrix_file='c1_8_1/distance_matrix.csv'):
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Make paths relative to script directory
    customers_file = os.path.join(script_dir, customers_file)
    vehicles_file = os.path.join(script_dir, vehicles_file)
    distance_matrix_file = os.path.join(script_dir, distance_matrix_file)
    
    customers_df = pd.read_csv(customers_file)
    vehicles_df = pd.read_csv(vehicles_file)
    distance_matrix_df = pd.read_csv(distance_matrix_file, index_col=0)
    distance_matrix_df = distance_matrix_df.apply(pd.to_numeric, errors='coerce')

    vehicles_df.set_index('vehicle_type', inplace=True)

    # Keep time windows in minutes (Solomon standard format)
    # No conversion needed - travel time = distance (1 distance unit = 1 minute)
    
    distance_matrix_array = distance_matrix_df.to_numpy(dtype=np.float32)
    
    # Map customer IDs to matrix indices (depot is index 0, customers start at 1)
    customer_addr_idx = customers_df['customer_id'].astype(np.int32).to_numpy()

    # Create compact arrays for fast checking
    def _num(col, default=0.0, scale=1.0, dtype=np.float32):
        vals = customers_df[col].fillna(default)
        if scale != 1.0:
            vals = vals * scale
        return vals.to_numpy(dtype)

    customer_arrays = {
        'demand': _num('demand', 0.0, 1.0, np.float32),
        'tw_start': _num('ready_time', 0.0, 1.0, np.float32),  # Keep in minutes
        'tw_end': _num('due_date', 0.0, 1.0, np.float32),      # Keep in minutes
        'service_time': _num('service_time', 0.0, 1.0, np.float32),  # Keep in minutes
    }

    return customers_df, vehicles_df, distance_matrix_df, distance_matrix_array, customer_addr_idx, customer_arrays

