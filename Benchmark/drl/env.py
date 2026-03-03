import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import os

# --- PATH SETUP ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import create_initial_solution, evaluate_solution
from utils.actions import build_actions, decode_action, NUM_ACTIONS
class VRPTWEnv(gym.Env):
    def __init__(self, customers_df, vehicles_df, dist_matrix, cust_addr_idx, cust_arrays, max_iterations):
        super(VRPTWEnv, self).__init__()
        
        # --- Data Loading ---
        self.customers_df = customers_df
        self.vehicles_df = vehicles_df
        self.dist_matrix = dist_matrix
        self.cust_addr_idx = cust_addr_idx
        self.cust_arrays = cust_arrays
        self.num_customers = len(customers_df['customer_id']) if isinstance(customers_df, dict) else len(customers_df)
        self.num_vehicles = int(vehicles_df['num_vehicles']) if isinstance(vehicles_df, dict) else int(vehicles_df.loc['Standard', 'num_vehicles'])

        # --- Actions (shared 30 composite actions) ---
        self.actions = build_actions()
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # --- Observation Space (21D: 17 core + 4 instance-structure features) ---
        state_dim = 21
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

        # --- RRT Params ---
        self.rrt_start_deviation = 0.10 
        self.max_iterations = max_iterations

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

        # --- Instance-Structure Features (computed once per reset, constant per episode) ---
        self.mean_tw_width_norm = 0.0
        self.tw_cv = 0.0
        self.demand_utilization_ratio = 0.0
        self.clustering_coeff = 0.0
        
        # --- Logging Stats (For TensorBoard) ---
        self.action_counts = np.zeros(NUM_ACTIONS)
        self.improvement_counts = np.zeros(NUM_ACTIONS)

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

        # --- Compute instance-structure features (constant for this episode) ---
        tw_start = self.cust_arrays['tw_start']
        tw_end   = self.cust_arrays['tw_end']
        tw_widths = tw_end - tw_start
        max_due = max(float(tw_end.max()), 1e-6)
        tw_mean = max(float(tw_widths.mean()), 1e-6)
        self.mean_tw_width_norm = float(tw_widths.mean()) / max_due
        self.tw_cv = float(tw_widths.std()) / tw_mean

        demands = self.cust_arrays['demand']
        capacity = self.vehicles_df['capacity'] if isinstance(self.vehicles_df, dict) else float(self.vehicles_df.loc['Standard', 'capacity'])
        self.demand_utilization_ratio = float(demands.sum()) / max(capacity * self.num_vehicles, 1e-6)

        x = self.customers_df['x_coord']
        y = self.customers_df['y_coord']
        coords = np.stack([x, y], axis=1)                          # (n, 2)
        diffs = coords[:, None, :] - coords[None, :, :]            # (n, n, 2)
        dists = np.sqrt((diffs ** 2).sum(axis=2))                  # (n, n)
        np.fill_diagonal(dists, np.inf)
        mean_nn_dist = float(dists.min(axis=1).mean())
        bbox_diag = max(np.sqrt((x.max()-x.min())**2 + (y.max()-y.min())**2), 1e-6)
        self.clustering_coeff = mean_nn_dist / float(bbox_diag)

        # Track initial solution as seen
        self.seen_solutions.add(tuple(sorted([tuple(r) for r in self.current_sol.routes])))
        
        return self._get_state(), {}

    def step(self, action_idx):
        self.iteration += 1
        self.action_counts[action_idx] += 1
        self.last_action = action_idx
        
        # Decode action: 20 = 2 destroy × 5 sizes × 2 repair
        self.last_destroy_op, self.last_bucket, self.last_repair = decode_action(action_idx)

        # 1. Execute Heuristics
        d_op, r_op, _label = self.actions[action_idx]
        destroyed = d_op(self.current_sol, distance_matrix_array=self.dist_matrix,
                         customer_addr_idx=self.cust_addr_idx, customer_arrays=self.cust_arrays)

        repaired = r_op(destroyed, distance_matrix_array=self.dist_matrix,
                        customer_addr_idx=self.cust_addr_idx, customer_arrays=self.cust_arrays,
                        vehicles_df=self.vehicles_df)
        
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

        # Capture prev best before update (needed for potential-based shaping)
        prev_best_cost = self.best_sol._cost

        if is_new_best:
            self.best_sol = repaired.copy()
            self.best_sol._cost = new_cost

        if accepted:
            self.current_sol = repaired
            self.last_distance = new_cost

        # Potential-based shaping: proportional to improvement in best, scale-invariant
        # Only non-zero when a new global best is found
        potential_shaping = (prev_best_cost - self.best_sol._cost) / max(self.initial_cost, 1e-6)

        # Exploration bonus: small reward for visiting novel solutions
        repaired_tuple = tuple(sorted([tuple(r) for r in repaired.routes]))
        is_repaired_unseen = repaired_tuple not in self.seen_solutions
        exploration_bonus = 0.1 if is_repaired_unseen else 0.0
        self.seen_solutions.add(repaired_tuple)
        self.is_unseen = 1.0 if is_repaired_unseen else 0.0

        # Base reward + shaping + exploration (5/3/1/0 flat tiers)
        if is_improvement:
            base_reward = 5.0 if is_new_best else 3.0
        elif accepted:
            base_reward = 1.0
        else:
            base_reward = 0.0
        reward = base_reward + potential_shaping + exploration_bonus

        # # Proportional reward only: scale shaping so 1% improvement ≈ reward 1.0 (DISABLED for 5310 comparison)
        # reward = potential_shaping * 100 + exploration_bonus

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
        Return 17D state vector representing the Hybrid State:
        [0]: Reduced Distance (improvement magnitude / initial_cost)
        [1]: Optimality Gap (Normalized % difference from Global Best)
        [2]: Current Cost (Scaled)
        [3]: Best Cost (Scaled)
        [4]: Temperature/Threshold (RRT allowance / initial_cost)
        [5]: Cost Trajectory (current - best / initial_cost)
        [6]: Stagnation (no_improvement_counter / max_iterations → [0, 1])
        [7]: Solution Changed (Binary flag)
        [8]: Unseen Solution (Binary flag)
        [9]: Delta Sign (Direction of last move)
        [10-12]: One-hot Destroy Operator (3 features: random, worst, cluster)
        [13]: Bucket Index (Normalized intensity)
        [14]: Repair Operator (Binary context)
        [15]: Remaining Budget (1.0 -> 0.0)
        [16]: Feasibility Pressure (Ratio of unassigned/dummy customers)
        [17]: Mean TW Width Norm (mean TW width / max due time — C1 vs C2 discriminator)
        [18]: TW CV (std / mean of TW widths — spread of time windows)
        [19]: Demand Utilization Ratio (total demand / total fleet capacity)
        [20]: Clustering Coeff (mean nearest-neighbour dist / bounding box diagonal — C vs R vs RC)
        """

        # --- 1. Core Metrics ---
        distance = self.current_sol._cost
        min_distance = max(self.best_sol._cost, 1e-6)

        optimality_gap = (distance - min_distance) / min_distance

        initial = max(self.initial_cost, 1e-6)
        dist_scaled = distance / initial
        min_dist_scaled = min_distance / initial

        reduced_dist = self.reduced_dist / initial                              # [0] normalized
        threshold_value = self.rrt_start_deviation * self.remaining_ratio * min_distance / initial  # [4] normalized
        cost_delta = (distance - min_distance) / initial                        # [5] normalized

        no_improvement = self.no_improvement_counter / self.max_iterations  # [0, 1]
        was_changed = self.was_changed
        unseen = self.is_unseen

        # --- 2. Action Context ---
        delta_sign = -1.0 if reduced_dist > 0 else 1.0

        # One-hot encoded destroy operators (3 features: random, worst, cluster)
        destroy_op_onehot = np.zeros(3, dtype=np.float32)
        if 0 <= self.last_destroy_op < 3:
            destroy_op_onehot[self.last_destroy_op] = 1.0

        bucket_norm = self.last_bucket / 4.0
        repair_norm = float(self.last_repair)

        # --- 3. Domain Features ---
        remaining_ratio = self.remaining_ratio

        dummy_route_len = len(self.current_sol.routes[-1])
        feasibility_pressure = dummy_route_len / max(self.num_customers, 1)

        # Combine into 21D vector
        state = np.array([
            reduced_dist,                    # [0]
            optimality_gap,                  # [1]
            dist_scaled,                     # [2]
            min_dist_scaled,                 # [3]
            threshold_value,                 # [4]
            cost_delta,                      # [5]
            no_improvement,                  # [6]
            was_changed,                     # [7]
            unseen,                          # [8]
            delta_sign,                      # [9]
            *destroy_op_onehot,              # [10-12]
            bucket_norm,                     # [13]
            repair_norm,                     # [14]
            remaining_ratio,                 # [15]
            feasibility_pressure,            # [16]
            self.mean_tw_width_norm,         # [17]
            self.tw_cv,                      # [18]
            self.demand_utilization_ratio,   # [19]
            self.clustering_coeff,           # [20]
        ], dtype=np.float32)

        return state