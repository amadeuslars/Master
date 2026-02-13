import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import os

# --- PATH SETUP ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import create_initial_solution, evaluate_solution, precompute_nearest_neighbors
from utils.operators import (
    random_removal, worst_removal, cluster_removal, shaw_removal, least_used_vehicle_removal,
    greedy_insertion, regret_insertion
)

class VRPTWEnv(gym.Env):
    def __init__(self, customers_df, vehicles_df, dist_matrix, cust_addr_idx, cust_arrays, seed=42):
        super(VRPTWEnv, self).__init__()
        
        # --- Data Loading ---
        self.customers_df = customers_df
        self.vehicles_df = vehicles_df
        self.dist_matrix = dist_matrix
        self.cust_addr_idx = cust_addr_idx
        self.cust_arrays = cust_arrays
        self.num_customers = len(customers_df['customer_id']) if isinstance(customers_df, dict) else len(customers_df)
        self.num_vehicles = int(vehicles_df['num_vehicles']) if isinstance(vehicles_df, dict) else int(vehicles_df.loc['Standard', 'num_vehicles'])
        
        self.neighbor_sets = precompute_nearest_neighbors(dist_matrix, num_neighbors=30)

        # --- Actions ---
        remove_buckets = [(0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40)]

        def bucketed(op, min_pct, max_pct):
            def _op(solution, **kwargs):
                low = max(1, int(self.num_customers * min_pct))
                high = max(low, int(self.num_customers * max_pct))
                n_remove = np.random.randint(low, high + 1)
                return op(solution, n_remove, **kwargs)
            return _op

        base_destroy_ops = [random_removal, worst_removal, cluster_removal, shaw_removal, least_used_vehicle_removal]
        self.destroy_ops = [bucketed(op, lo, hi) for op in base_destroy_ops for lo, hi in remove_buckets]
        self.repair_ops = [greedy_insertion, regret_insertion]
        self.action_pairs = [(d, r) for d in self.destroy_ops for r in self.repair_ops]
        self.action_space = spaces.Discrete(len(self.action_pairs))
        
        # --- Observation Space (19D: 9 core + 1 sign + 5 one-hot destroy + 1 bucket + 1 repair + 2 domain) ---
        state_dim = 19
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

        # --- RRT Params ---
        self.rrt_start_deviation = 0.20 
        self.max_iterations = 1000

        # --- Internal State ---
        self.current_sol = None
        self.best_sol = None
        self.iteration = 0
        self.no_improvement_counter = 0
        
        # --- State Tracking ---
        self.last_action = 0
        self.last_distance = 0
        self.was_changed = 0
        self.seen_solutions = set()
        self.T = 0.0
        self.cs = 0.0
        self.reduced_dist = 0.0
        self.last_destroy_op = 0
        self.last_bucket = 0
        self.last_repair = 0
        self.remaining_ratio = 1.0
        self.initial_cost = 0.0
        self.is_unseen = 0.0  # Track before adding to seen set
        
        # --- Logging Stats (For TensorBoard) ---
        self.action_counts = np.zeros(len(self.action_pairs))
        self.improvement_counts = np.zeros(len(self.action_pairs))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        self.iteration = 0
        self.action_counts.fill(0)
        self.improvement_counts.fill(0)
        self.no_improvement_counter = 0
        self.last_action = 0
        self.was_changed = 0
        self.T = 0.0
        self.cs = 0.0
        self.reduced_dist = 0.0
        self.seen_solutions = set()
        self.last_destroy_op = 0
        self.last_bucket = 0
        self.last_repair = 0
        self.remaining_ratio = 1.0
        self.is_unseen = 0.0
        
        self.current_sol = create_initial_solution(self.num_customers, self.num_vehicles)
        evaluate_solution(self.current_sol, self.dist_matrix, self.cust_addr_idx)
        self.last_distance = self.current_sol._cost
        self.initial_cost = self.current_sol._cost
        
        self.best_sol = self.current_sol.copy()
        self.best_sol._cost = self.current_sol._cost
        
        # Track initial solution as seen
        self.seen_solutions.add(tuple(sorted([tuple(r) for r in self.current_sol.routes])))
        
        return self._get_state(), {}

    def step(self, action_idx):
        self.iteration += 1
        self.action_counts[action_idx] += 1
        self.last_action = action_idx
        
        # Decode action: 50 = 5 destroy × 5 buckets × 2 repair
        self.last_repair = action_idx % 2
        bucket_and_destroy = action_idx // 2
        self.last_bucket = bucket_and_destroy % 5
        self.last_destroy_op = bucket_and_destroy // 5
        
        # 1. Execute Heuristics
        d_op, r_op = self.action_pairs[action_idx]
        destroyed = d_op(self.current_sol, distance_matrix_array=self.dist_matrix, 
                         customer_addr_idx=self.cust_addr_idx, customer_arrays=self.cust_arrays)
        
        repaired = r_op(destroyed, distance_matrix_array=self.dist_matrix, 
                        customer_addr_idx=self.cust_addr_idx, customer_arrays=self.cust_arrays, 
                        vehicles_df=self.vehicles_df, neighbor_sets=self.neighbor_sets)
        
        new_cost = evaluate_solution(repaired, self.dist_matrix, self.cust_addr_idx)
        prev_cost = self.current_sol._cost
        
        # Track if solution changed
        self.was_changed = 1 if new_cost != prev_cost else 0
        
        # Track reduced distance
        self.reduced_dist = prev_cost - new_cost if new_cost < prev_cost else 0.0
        
        # 2. Acceptance Criteria (RRT)
        self.remaining_ratio = max(0, (self.max_iterations - self.iteration) / self.max_iterations)
        threshold_value = self.rrt_start_deviation * self.remaining_ratio * self.best_sol._cost
        accepted = new_cost < (self.best_sol._cost + threshold_value)

        # 3. Reward & Improvement Tracking
        is_improvement = new_cost < prev_cost
        is_new_best = new_cost < self.best_sol._cost

        if is_improvement:
            self.no_improvement_counter = 0
            self.improvement_counts[action_idx] += 1
        else:
            self.no_improvement_counter += 1

        if is_new_best:
            self.best_sol = repaired.copy()
            self.best_sol._cost = new_cost

        if accepted:
            self.current_sol = repaired
            self.last_distance = new_cost

        if is_improvement:
            reward = 5.0 if is_new_best else 3.0
        elif accepted:
            reward = 1.0
        else:
            reward = 0.0

        # Track if this is a new/unseen solution (BEFORE adding to set)
        sol_tuple = tuple(sorted([tuple(r) for r in self.current_sol.routes]))
        self.is_unseen = 1.0 if sol_tuple not in self.seen_solutions else 0.0
        self.seen_solutions.add(sol_tuple)

        # 4. Termination
        terminated = False
        truncated = self.iteration >= self.max_iterations
        
        # Pass detailed stats in info
        info = {
            'best_cost': self.best_sol._cost,
            'initial_cost': self.initial_cost,
            'action_counts': self.action_counts,
            'improvement_counts': self.improvement_counts
        }
        
        return self._get_state(), reward, terminated, truncated, info

    def _get_state(self):
        """
        Return 19D state vector representing the Hybrid State:
        [0]: Reduced Distance (Improvement magnitude since last step)
        [1]: Optimality Gap (Normalized % difference from Global Best)
        [2]: Current Cost (Scaled)
        [3]: Best Cost (Scaled)
        [4]: Temperature/Threshold (Current RRT allowance)
        [5]: Cost Trajectory (Raw differential)
        [6]: Stagnation (No improvement counter)
        [7]: Solution Changed (Binary flag)
        [8]: Unseen Solution (Binary flag)
        [9]: Delta Sign (Direction of last move)
        [10-14]: One-hot Destroy Operator (Previous action context)
        [15]: Bucket Index (Normalized intensity)
        [16]: Repair Operator (Binary context)
        [17]: Remaining Budget (1.0 -> 0.0)
        [18]: Feasibility Pressure (Ratio of unassigned/dummy customers)
        """
        
        # --- 1. Core Metrics ---
        reduced_dist = self.reduced_dist
        distance = self.current_sol._cost
        min_distance = max(self.best_sol._cost, 1e-6) # Avoid div/0
        
        # Thesis Consistency: "Optimality Gap" should be normalized
        optimality_gap = (distance - min_distance) / min_distance
        
        # Normalize by initial cost for instance-agnostic representation
        initial = max(self.initial_cost, 1e-6)  # Avoid div/0
        dist_scaled = distance / initial
        min_dist_scaled = min_distance / initial
        
        # RRT threshold value (not temperature - actual acceptance slack)
        threshold_value = self.rrt_start_deviation * self.remaining_ratio * min_distance
        
        # Cost Trajectory: use raw delta (not duplicate of reduced_dist)
        cost_delta = distance - min_distance  # Absolute gap
        
        no_improvement = self.no_improvement_counter
        was_changed = self.was_changed
        
        # Use pre-stored unseen status
        unseen = self.is_unseen
        
        # --- 2. Action Context ---
        delta_sign = -1.0 if reduced_dist > 0 else 1.0
        
        # One-hot encoded destroy operators (5 features)
        destroy_op_onehot = np.zeros(5, dtype=np.float32)
        if 0 <= self.last_destroy_op < 5:
            destroy_op_onehot[self.last_destroy_op] = 1.0
            
        # Bucket feature (normalized 0-4 -> 0.0-1.0)
        bucket_norm = self.last_bucket / 4.0
        
        # Repair feature (0 or 1)
        repair_norm = float(self.last_repair)
        
        # --- 3. Domain Features (Thesis Consistency) ---
        
        # Remaining Budget (Thesis: "1.0 - t/T_max")
        remaining_ratio = self.remaining_ratio
        
        # NEW: Feasibility Pressure
        # Calculates ratio of customers in the dummy vehicle (route index -1).
        # High pressure = many customers could not be inserted feasibly -> Agent should switch strategies.
        dummy_route_len = len(self.current_sol.routes[-1])
        feasibility_pressure = dummy_route_len / max(self.num_customers, 1)
        
        # Combine into 19D vector
        state = np.array([
            reduced_dist,       # [0] - improvement this step
            optimality_gap,     # [1] - normalized gap
            dist_scaled,        # [2] - current/initial
            min_dist_scaled,    # [3] - best/initial
            threshold_value,    # [4] - RRT acceptance slack
            cost_delta,         # [5] - absolute gap from best
            no_improvement,     # [6]
            was_changed,        # [7]
            unseen,             # [8]
            delta_sign,         # [9]
            *destroy_op_onehot, # [10-14]
            bucket_norm,        # [15]
            repair_norm,        # [16]
            remaining_ratio,    # [17]
            feasibility_pressure # [18]
        ], dtype=np.float32)
        
        return state