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
        self.num_customers = len(customers_df)
        self.num_vehicles = int(vehicles_df.loc['Standard', 'num_vehicles'])
        
        self.neighbor_sets = precompute_nearest_neighbors(dist_matrix, num_neighbors=15)

        # --- Actions ---
        self.destroy_ops = [random_removal, worst_removal, cluster_removal, shaw_removal, least_used_vehicle_removal]
        self.repair_ops = [greedy_insertion, regret_insertion]
        self.action_pairs = [(d, r) for d in self.destroy_ops for r in self.repair_ops]
        self.action_space = spaces.Discrete(len(self.action_pairs))
        
        # --- Observation Space ---
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)

        # --- RRT Params ---
        self.rrt_start_deviation = 0.10 
        self.max_iterations = 200 

        # --- Internal State ---
        self.current_sol = None
        self.best_sol = None
        self.iteration = 0
        self.no_improvement_counter = 0
        
        # --- Logging Stats (For TensorBoard) ---
        self.action_counts = np.zeros(len(self.action_pairs))
        self.improvement_counts = np.zeros(len(self.action_pairs))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        self.iteration = 0
        self.no_improvement_counter = 0
        self.action_counts.fill(0)
        self.improvement_counts.fill(0)
        
        self.current_sol = create_initial_solution(self.num_customers, self.num_vehicles)
        evaluate_solution(self.current_sol, self.dist_matrix, self.cust_addr_idx)
        
        self.best_sol = self.current_sol.copy()
        self.best_sol._cost = self.current_sol._cost
        
        return self._get_state(), {}

    def step(self, action_idx):
        self.iteration += 1
        self.action_counts[action_idx] += 1  # Log usage
        
        # 1. Execute Heuristics
        d_op, r_op = self.action_pairs[action_idx]
        n_remove = np.random.randint(int(self.num_customers * 0.1), int(self.num_customers * 0.3) + 1)
        
        destroyed = d_op(self.current_sol, n_remove, distance_matrix_array=self.dist_matrix, 
                         customer_addr_idx=self.cust_addr_idx, customer_arrays=self.cust_arrays)
        
        repaired = r_op(destroyed, distance_matrix_array=self.dist_matrix, 
                        customer_addr_idx=self.cust_addr_idx, customer_arrays=self.cust_arrays, 
                        vehicles_df=self.vehicles_df, neighbor_sets=self.neighbor_sets)
        
        new_cost = evaluate_solution(repaired, self.dist_matrix, self.cust_addr_idx)
        prev_cost = self.current_sol._cost
        
        # 2. Acceptance Criteria (RRT)
        accepted = False
        reward = 0.0
        
        remaining_ratio = max(0, (self.max_iterations - self.iteration) / self.max_iterations)
        threshold_value = self.rrt_start_deviation * remaining_ratio * self.best_sol._cost
        
        if new_cost < self.best_sol._cost + threshold_value:
            accepted = True
        
        # 3. Reward & Improvement Tracking
        if new_cost < prev_cost:
            self.no_improvement_counter = 0
            self.improvement_counts[action_idx] += 1 # Log success
            reward = 1.0
        else:
            self.no_improvement_counter += 1
            reward = -0.1
            
        if new_cost < self.best_sol._cost:
            self.best_sol = repaired.copy()
            self.best_sol._cost = new_cost
            reward = 5.0
            accepted = True 
            
        if accepted:
            self.current_sol = repaired

        # 4. Termination
        terminated = False
        truncated = self.iteration >= self.max_iterations
        
        # Pass detailed stats in info
        info = {
            'best_cost': self.best_sol._cost,
            'initial_cost': self.current_sol._cost if self.iteration == 1 else 0, # Approximate
            'action_counts': self.action_counts,
            'improvement_counts': self.improvement_counts
        }
        
        return self._get_state(), reward, terminated, truncated, info

    def _get_state(self):
        gap = (self.current_sol._cost - self.best_sol._cost) / (self.best_sol._cost + 1e-5)
        progress = self.iteration / self.max_iterations
        stagnation = min(self.no_improvement_counter / 50.0, 1.0)
        
        total_demand = self.cust_arrays['demand'].sum()
        num_vehicles_used = len([r for r in self.current_sol.routes[:-1] if r])
        capacity = self.vehicles_df.iloc[0]['capacity'] if num_vehicles_used > 0 else 1.0
        avg_load = total_demand / (num_vehicles_used * capacity) if num_vehicles_used > 0 else 1.0
            
        unassigned_count = len(self.current_sol.routes[-1])
        unassigned_feat = unassigned_count / self.num_customers
        remaining_ratio = max(0, (self.max_iterations - self.iteration) / self.max_iterations)
        
        return np.array([gap, progress, stagnation, avg_load, unassigned_feat, remaining_ratio], dtype=np.float32)