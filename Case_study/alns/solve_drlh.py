"""
Solve the Case Study VRPTW using the pre-trained DRLH agent (5310_all_files_agent).

The agent was trained on Benchmark instances (Homberger/Solomon) with a 21D state
and 30 composite actions (3 destroy × 5 sizes × 2 repair). This script adapts the
Case Study's multi-trip, shift-aware ALNS to that same state/action interface so
the pre-trained PPO policy can drive operator selection.

Usage:
    python Case_study/alns/solve_drlh.py
"""

import sys
import os
import time
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import torch

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'Case_study'))

from utils.utils import load_vrp_data, generate_initial_solution, evaluate_solution

try:
    from utils.operators_cy import (
        random_removal, worst_removal, cluster_removal,
        greedy_insertion, regret_insertion,
        two_opt_local_search, or_opt_local_search,
    )
    _USING_CYTHON = True
except ImportError:
    from utils.operators import (
        random_removal, worst_removal, cluster_removal,
        greedy_insertion, regret_insertion,
        two_opt_local_search, or_opt_local_search,
    )
    _USING_CYTHON = False

from utils.feasibility import compute_route_schedule, get_earliest_departure

# ── Configuration ──
MAX_ITERATIONS = 10000
MODEL_PATH = os.path.join(project_root, 'logs', '5310_all_files_agent', 'ppo_vrptw_final.zip')
CUSTOMERS_FILE = 'Case_study/data/customers_alesund_sula_tue.csv'
DELIVERY_DAY = 'tue'

# ── Action space: 30 = 3 destroy × 5 sizes × 2 repair ──
REMOVAL_SIZES = [
    ('xs', 2, 5),
    ('sm', 5, 10),
    ('md', 10, 20),
    ('lg', 20, 30),
    ('xl', 30, 40),
]

DESTROY_OPS = [
    ('random', random_removal),
    ('worst', worst_removal),
    ('cluster', cluster_removal),
]

REPAIR_OPS = [
    ('greedy', greedy_insertion),
    ('regret', regret_insertion),
]

NUM_ACTIONS = len(DESTROY_OPS) * len(REMOVAL_SIZES) * len(REPAIR_OPS)  # 30


def decode_action(idx):
    r_idx = idx % 2
    s_idx = (idx // 2) % 5
    d_idx = idx // 10
    return d_idx, s_idx, r_idx


class CaseStudyDRLHEnv(gym.Env):
    """
    Gymnasium environment that wraps the Case Study ALNS so the pre-trained
    DRLH agent can drive operator selection.

    The 21D state vector and 30-action space match the Benchmark VRPTWEnv exactly.
    The underlying operators are the Case Study's multi-trip, shift-aware versions.
    """

    def __init__(self, customers_dict, vehicles_dict, vehicle_names,
                 time_matrix_array, depot_idx, addr_idx, customer_arrays,
                 compatible_ppls_set, sorted_vehicle_names, max_iterations):
        super().__init__()

        self.customers_dict = customers_dict
        self.vehicles_dict = vehicles_dict
        self.vehicle_names = vehicle_names
        self.time_matrix_array = time_matrix_array
        self.depot_idx = depot_idx
        self.addr_idx = addr_idx
        self.customer_arrays = customer_arrays
        self.compatible_ppls_set = compatible_ppls_set
        self.sorted_vehicle_names = sorted_vehicle_names
        self.num_customers = len(customers_dict['customer_id'])
        self.max_iterations = max_iterations

        # Action / observation spaces (must match trained model)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )

        # RRT (must match training: benchmark env uses 0.10)
        self.rrt_start_deviation = 0.10

        # Internal state
        self.current_sol = None
        self.best_sol = None
        self.iteration = 0
        self.no_improvement_counter = 0
        self.last_action = 0
        self.last_distance = 0
        self.was_changed = 0
        self.seen_solutions = set()
        self.reduced_dist = 0.0
        self.last_destroy_op = 0
        self.last_bucket = 0
        self.last_repair = 0
        self.remaining_ratio = 1.0
        self.initial_cost = 0.0
        self.is_unseen = 0.0

        # Instance-structure features
        self.mean_tw_width_norm = 0.0
        self.tw_cv = 0.0
        self.demand_utilization_ratio = 0.0
        self.clustering_coeff = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.iteration = 0
        self.no_improvement_counter = 0
        self.last_action = 0
        self.was_changed = 0
        self.reduced_dist = 0.0
        self.seen_solutions = set()
        self.last_destroy_op = 0
        self.last_bucket = 0
        self.last_repair = 0
        self.remaining_ratio = 1.0
        self.is_unseen = 0.0

        # Create initial solution (all customers in dummy)
        self.current_sol = generate_initial_solution(
            self.customers_dict, vehicle_names=self.sorted_vehicle_names
        )
        self.initial_cost = evaluate_solution(
            self.current_sol, self.addr_idx, self.time_matrix_array, self.depot_idx
        )
        self.last_distance = self.initial_cost

        self.best_sol = self.current_sol.copy()
        self.best_sol.cost = self.initial_cost

        # Compute instance-structure features (constant per episode)
        tw_start = self.customer_arrays['tw_start']
        tw_end = self.customer_arrays['tw_end']
        tw_widths = tw_end - tw_start
        max_due = max(float(tw_end.max()), 1e-6)
        tw_mean = max(float(tw_widths.mean()), 1e-6)
        self.mean_tw_width_norm = float(tw_widths.mean()) / max_due
        self.tw_cv = float(tw_widths.std()) / tw_mean

        demands = self.customer_arrays['demand']
        # Approximate total fleet capacity (sum of all vehicle PPL × 4 trips)
        total_cap = sum(
            self.vehicles_dict[v]['PPL total']
            for v in self.sorted_vehicle_names if v != 'dummy'
        )
        self.demand_utilization_ratio = float(demands.sum()) / max(total_cap, 1e-6)

        # Clustering coefficient from lat/lon (not x/y like benchmark, but same idea)
        from pandas import read_csv
        cust_df = read_csv(CUSTOMERS_FILE)
        cust_df = cust_df[cust_df['delivery_day'].eq(DELIVERY_DAY)].reset_index(drop=True)
        lats = cust_df['latitude'].to_numpy(dtype=np.float64)
        lons = cust_df['longitude'].to_numpy(dtype=np.float64)
        coords = np.stack([lats, lons], axis=1)
        diffs = coords[:, None, :] - coords[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        mean_nn_dist = float(dists.min(axis=1).mean())
        bbox_diag = max(
            np.sqrt((lats.max() - lats.min())**2 + (lons.max() - lons.min())**2),
            1e-6
        )
        self.clustering_coeff = mean_nn_dist / bbox_diag

        self.seen_solutions.add(
            tuple(sorted([tuple(r) for r in self.current_sol.routes]))
        )

        return self._get_state(), {}

    def step(self, action_idx):
        self.iteration += 1
        self.last_action = action_idx
        d_idx, s_idx, r_idx = decode_action(action_idx)
        self.last_destroy_op = d_idx
        self.last_bucket = s_idx
        self.last_repair = r_idx

        # Determine removal size
        _, lo, hi = REMOVAL_SIZES[s_idx]
        n_remove = random.randint(lo, min(hi, self.num_customers))

        # Execute destroy
        d_name, d_op = DESTROY_OPS[d_idx]
        destroyed = d_op(
            self.current_sol, n_remove,
            time_matrix_array=self.time_matrix_array,
            customer_addr_idx=self.addr_idx,
            customer_arrays=self.customer_arrays,
            depot_idx=self.depot_idx,
        )

        # Execute repair
        r_name, r_op = REPAIR_OPS[r_idx]
        repaired = r_op(
            destroyed,
            self.time_matrix_array,
            self.addr_idx,
            self.customer_arrays,
            self.vehicles_dict,
            None,
            depot_idx=self.depot_idx,
            temperature=1.0,
            compatible_ppls_set=self.compatible_ppls_set,
        )

        new_cost = evaluate_solution(
            repaired, self.addr_idx, self.time_matrix_array, self.depot_idx
        )
        prev_cost = self.current_sol.cost

        self.was_changed = 1 if new_cost != prev_cost else 0
        self.reduced_dist = prev_cost - new_cost if new_cost < prev_cost else 0.0

        # RRT acceptance
        self.remaining_ratio = max(0, (self.max_iterations - self.iteration) / self.max_iterations)
        threshold_value = self.rrt_start_deviation * self.remaining_ratio * self.best_sol.cost
        accepted = new_cost < (self.best_sol.cost + threshold_value)

        is_improvement = new_cost < prev_cost
        is_new_best = new_cost < self.best_sol.cost

        if is_improvement:
            self.no_improvement_counter = 0
        else:
            self.no_improvement_counter += 1

        prev_best_cost = self.best_sol.cost

        # Update best regardless of RRT acceptance (matches benchmark env)
        if is_new_best:
            self.best_sol = repaired.copy()
            self.best_sol.cost = new_cost

        if accepted:
            # Local search disabled for fair comparison (no LS in benchmark training)
            # repaired = two_opt_local_search(
            #     repaired, self.time_matrix_array, self.addr_idx,
            #     self.customer_arrays, depot_idx=self.depot_idx
            # )
            # repaired = or_opt_local_search(
            #     repaired, self.time_matrix_array, self.addr_idx,
            #     self.customer_arrays, depot_idx=self.depot_idx
            # )
            # new_cost = evaluate_solution(
            #     repaired, self.addr_idx, self.time_matrix_array, self.depot_idx
            # )
            self.current_sol = repaired
            self.last_distance = new_cost

        # Reward (matching benchmark agent's reward structure)
        potential_shaping = (prev_best_cost - self.best_sol.cost) / max(self.initial_cost, 1e-6)
        repaired_tuple = tuple(sorted([tuple(r) for r in repaired.routes]))
        is_repaired_unseen = repaired_tuple not in self.seen_solutions
        exploration_bonus = 0.1 if is_repaired_unseen else 0.0
        self.seen_solutions.add(repaired_tuple)
        self.is_unseen = 1.0 if is_repaired_unseen else 0.0

        if is_improvement:
            base_reward = 5.0 if new_cost < prev_best_cost else 3.0
        elif accepted:
            base_reward = 1.0
        else:
            base_reward = 0.0
        reward = base_reward + potential_shaping + exploration_bonus

        terminated = False
        truncated = self.iteration >= self.max_iterations

        info = {
            'best_cost': self.best_sol.cost,
            'initial_cost': self.initial_cost,
        }

        return self._get_state(), reward, terminated, truncated, info

    def _get_state(self):
        """21D state vector matching the benchmark VRPTWEnv exactly."""
        distance = self.current_sol.cost
        min_distance = max(self.best_sol.cost, 1e-6)
        optimality_gap = (distance - min_distance) / min_distance

        initial = max(self.initial_cost, 1e-6)
        dist_scaled = distance / initial
        min_dist_scaled = min_distance / initial

        reduced_dist = self.reduced_dist / initial
        threshold_value = self.rrt_start_deviation * self.remaining_ratio * min_distance / initial
        cost_delta = (distance - min_distance) / initial
        no_improvement = self.no_improvement_counter / self.max_iterations
        was_changed = self.was_changed
        unseen = self.is_unseen

        delta_sign = -1.0 if reduced_dist > 0 else 1.0
        destroy_op_onehot = np.zeros(3, dtype=np.float32)
        if 0 <= self.last_destroy_op < 3:
            destroy_op_onehot[self.last_destroy_op] = 1.0
        bucket_norm = self.last_bucket / 4.0
        repair_norm = float(self.last_repair)
        remaining_ratio = self.remaining_ratio

        dummy_route_len = len(self.current_sol.routes[-1])
        feasibility_pressure = dummy_route_len / max(self.num_customers, 1)

        state = np.array([
            reduced_dist,
            optimality_gap,
            dist_scaled,
            min_dist_scaled,
            threshold_value,
            cost_delta,
            no_improvement,
            was_changed,
            unseen,
            delta_sign,
            *destroy_op_onehot,
            bucket_norm,
            repair_norm,
            remaining_ratio,
            feasibility_pressure,
            self.mean_tw_width_norm,
            self.tw_cv,
            self.demand_utilization_ratio,
            self.clustering_coeff,
        ], dtype=np.float32)

        return state


def _fmt_time(h):
    hh = int(h)
    mm = int((h - hh) * 60)
    return f"{hh:02d}:{mm:02d}"


def solve_case_study(customers_file=CUSTOMERS_FILE, delivery_day=DELIVERY_DAY,
                     model_path=MODEL_PATH, max_iterations=MAX_ITERATIONS):
    """Run the DRLH agent on the case study instance."""
    print(f"[DRLH] Backend: {'Cython' if _USING_CYTHON else 'Python'}")
    print(f"[DRLH] Model: {model_path}")
    print(f"[DRLH] Customers: {customers_file}, day={delivery_day}")
    print(f"[DRLH] Max iterations: {max_iterations}")

    # Load data
    (customers_dict, vehicles_dict, vehicle_names,
     time_matrix_array, _, depot_idx, addr_idx, customer_arrays) = load_vrp_data(
        delivery_day=delivery_day, customers_file=customers_file
    )

    real_vehicle_indices = [i for i, name in enumerate(vehicle_names) if name != 'dummy']
    real_vehicles_sorted = sorted(
        real_vehicle_indices,
        key=lambda i: vehicles_dict[vehicle_names[i]]['PPL total'],
        reverse=True
    )
    sorted_vehicle_names = [vehicle_names[i] for i in real_vehicles_sorted] + ['dummy']

    compatible_ppls_set = set()
    for v_name in ['small', 'medium-small', 'medium']:
        if v_name in vehicles_dict:
            compatible_ppls_set.add(vehicles_dict[v_name]['PPL total'])

    num_customers = len(customers_dict['customer_id'])

    # Create environment
    env = CaseStudyDRLHEnv(
        customers_dict, vehicles_dict, vehicle_names,
        time_matrix_array, depot_idx, addr_idx, customer_arrays,
        compatible_ppls_set, sorted_vehicle_names, max_iterations
    )

    # Load trained model
    model = PPO.load(model_path, env=env)

    # Solve
    obs, _ = env.reset()
    done = False
    step = 0
    prev_best = env.best_sol.cost
    print(f"\nInitial cost: {prev_best:.2f} ({num_customers} customers)")

    destroy_names = ['random', 'worst', 'cluster']
    bucket_names = ['xs(2-5)', 'sm(5-10)', 'md(10-20)', 'lg(20-30)', 'xl(30-40)']
    repair_names = ['greedy', 'regret']

    # Build action labels
    action_labels = []
    for d in ['random', 'worst', 'cluster']:
        for s in ['xs', 'sm', 'md', 'lg', 'xl']:
            for r in ['greedy', 'regret']:
                action_labels.append(f"{d}_{s}_{r}")

    # History logging
    history = {
        'iterations': [],
        'actions': [],
        'costs': [],
        'policy_probs': [],
        'algorithm': 'DRLH',
        'action_labels': action_labels,
    }

    t0 = time.perf_counter()
    while not done:
        step += 1

        # Extract policy probabilities before predict
        obs_tensor = torch.as_tensor(obs).float().unsqueeze(0).to(model.policy.device)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.cpu().numpy()[0]  # shape (30,)

        action, _ = model.predict(obs, deterministic=True)

        d_idx, s_idx, r_idx = decode_action(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        current_best = env.best_sol.cost

        # Log history
        history['iterations'].append(step - 1)
        history['actions'].append(int(action))
        history['costs'].append(current_best)
        history['policy_probs'].append(probs.copy())

        if current_best < prev_best:
            improvement = prev_best - current_best
            assigned = sum(
                len(r) for r, v in zip(env.best_sol.routes, env.best_sol.vehicles)
                if v != 'dummy'
            )
            print(
                f"  [Step {step}] New best: {current_best:.2f} (↓{improvement:.2f}) "
                f"Assigned: {assigned}/{num_customers} "
                f"[{destroy_names[d_idx]}|{bucket_names[s_idx]}|{repair_names[r_idx]}]"
            )
            prev_best = current_best

        if step % 100 == 0:
            print(f"  [Step {step}] Best: {current_best:.2f}")

    elapsed = time.perf_counter() - t0

    # Final results
    best_sol = env.best_sol
    num_assigned = sum(
        len(r) for r, v in zip(best_sol.routes, best_sol.vehicles) if v != 'dummy'
    )
    num_unassigned = len(best_sol.routes[-1])
    num_trips = sum(
        1 for r, v in zip(best_sol.routes, best_sol.vehicles) if r and v != 'dummy'
    )
    used_vehicles = set()
    for i, (route, veh) in enumerate(zip(best_sol.routes, best_sol.vehicles)):
        if route and veh != 'dummy':
            meta = best_sol.route_meta[i]
            if meta:
                used_vehicles.add(meta['vehicle_idx'])

    print(f"\n{'='*50}")
    print(f"DRLH SOLUTION — {delivery_day.upper()}")
    print(f"{'='*50}")
    print(f"Best Cost: {best_sol.cost:.2f} hours ({best_sol.cost * 60:.0f} min)")
    print(f"Assigned: {num_assigned}/{num_customers} | Unassigned: {num_unassigned}")
    print(f"Vehicles used: {len(used_vehicles)} | Active trips: {num_trips}")
    print(f"Time: {elapsed:.1f}s ({elapsed/max_iterations*1000:.1f}ms/iter)")

    # Print schedule
    cids = customers_dict['customer_id']
    for i, (route, veh) in enumerate(zip(best_sol.routes, best_sol.vehicles)):
        if not route or veh == 'dummy':
            continue
        meta = best_sol.route_meta[i]
        earliest_dep = get_earliest_departure(
            best_sol, i, time_matrix_array, addr_idx, customer_arrays, depot_idx
        )
        lunch_pos = best_sol.lunch_breaks[i]
        events = compute_route_schedule(
            route, time_matrix_array, addr_idx, customer_arrays,
            depot_idx, earliest_dep, lunch_pos
        )
        label = f"{veh} T{meta['trip']}" if meta else veh
        print(f"\n  === {label} ({len(route)} stops) ===")
        for ev in events:
            cust_label = ""
            if ev['customer'] is not None:
                cust_label = f" [#{int(cids[ev['customer'] - 1])}]"
            print(f"    {_fmt_time(ev['time'])}  {ev['details']}{cust_label}")
        if events:
            last_time = events[-1]['time']
            print(f"    -- Done at {_fmt_time(last_time)} --")

    if best_sol.routes[-1]:
        unassigned_kundenr = [int(cids[c - 1]) for c in best_sol.routes[-1]]
        print(f"\n  Unassigned ({num_unassigned}): {unassigned_kundenr}")

    # Convert history to numpy arrays
    history['policy_probs'] = np.array(history['policy_probs'])
    history['actions'] = np.array(history['actions'])
    history['costs'] = np.array(history['costs'])

    return best_sol, history


if __name__ == '__main__':
    sol, history = solve_case_study()
