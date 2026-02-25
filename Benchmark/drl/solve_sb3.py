"""
Solve VRPTW instances using a trained SB3 PPO model.
"""
import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import time

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from Benchmark.drl.env import VRPTWEnv
from Benchmark.utils.utils import load_raw_solomon_data

# Configuration
MAX_ITERATIONS = 5000

def solve_with_sb3(instance_path, model_path):
    """
    Solve a VRPTW instance using a trained SB3 model.
    
    Args:
        instance_path: Path to instance .txt file
        model_path: Path to saved model (.zip file)
        deterministic: If True, use argmax action; if False, sample from policy
    """
    print(f"--- Loading instance: {instance_path} ---")
    
    if not os.path.exists(instance_path):
        print(f"Error: Instance not found at {instance_path}")
        return None
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None
    
    # Load data
    data = load_raw_solomon_data(instance_path)
    customers_df, vehicles_df, dist_matrix, cust_addr_idx, cust_arrays = data
    
    # Create environment
    env = VRPTWEnv(customers_df, vehicles_df, dist_matrix, cust_addr_idx, cust_arrays, MAX_ITERATIONS)
    
    # Load trained model
    print(f"--- Loading model: {model_path} ---")
    model = PPO.load(model_path, env=env, deterministic = True)
    
    # Run episode with tracking
    print("--- Solving with trained agent ---")
    obs, _ = env.reset()
    done = False
    step = 0
    prev_best = env.best_sol._cost
    print(f"  Initial solution: {prev_best:.2f}")
    
    # Tracking arrays
    cost_history = []
    action_history = []

    # Decode action to operator info
    # Action space: 20 actions = 2 destroy ops × 5 sizes × 2 repair ops
    destroy_ops = ['random', 'worst', 'cluster']
    buckets = ['xs(2-5)', 'sm(5-10)', 'md(10-20)', 'lg(20-30)', 'xl(30-40)']
    repair_ops = ['greedy', 'regret']

    while not done:
        step += 1
        action, _states = model.predict(obs)

        # Decode action (MUST match env.py / actions.py exactly)
        repair_idx = action % 2
        bucket_idx = (action // 2) % 5
        destroy_op_idx = action // 10
        
        action_history.append(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Track cost
        current_best = env.best_sol._cost
        cost_history.append(current_best)
        
        # Check if new best found
        if current_best < prev_best:
            improvement = prev_best - current_best
            print(f"  [Step {step}] New best: {current_best:.2f} (↓{improvement:.2f}) "
                  f"[{destroy_ops[destroy_op_idx]}|{buckets[bucket_idx]}|{repair_ops[repair_idx]}]")
            prev_best = current_best
            best_found_at = step
        
        # Progress update every 100 steps
        if step % 100 == 0:
            print(f"  [Step {step}] Current best: {current_best:.2f}")
    
    # Print results
    print("\n" + "="*50)
    print(f"SOLUTION: {os.path.basename(instance_path)}")
    print("="*50)
    print(f"Best Cost: {env.best_sol._cost:.2f}")
    print(f"Total Steps: {step}")
    
    print("\nRoutes:")
    vehicle_count = 0
    for i, route in enumerate(env.best_sol.routes[:-1]):  # Skip dummy vehicle
        if route:
            vehicle_count += 1
            load = sum(cust_arrays['demand'][c-1] for c in route)
            print(f"  Vehicle {vehicle_count}: {route} (load: {load})")
    
    print(f"\nVehicles Used: {vehicle_count}")
    print("="*50)
    
    # Create plots
    # plot_results(action_history, cost_history, destroy_ops, buckets, repair_ops, 
    #              os.path.basename(instance_path))
    
    return env.best_sol._cost, env.best_sol, best_found_at


def plot_results(action_history, cost_history, destroy_ops, buckets, repair_ops, instance_name):
    """Create visualizations of operator usage and convergence."""
    
    # Decode all actions
    destroy_counts = np.zeros(len(destroy_ops))
    bucket_counts = np.zeros(len(buckets))
    repair_counts = np.zeros(len(repair_ops))
    
    for action in action_history:
        repair_idx = action % 2
        bucket_idx = (action // 2) % 5
        destroy_op_idx = action // 10
        
        destroy_counts[destroy_op_idx] += 1
        bucket_counts[bucket_idx] += 1
        repair_counts[repair_idx] += 1
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Operator Analysis - {instance_name}', fontsize=16, fontweight='bold')
    
    # 1. Convergence plot
    ax = axes[0, 0]
    ax.plot(cost_history, linewidth=2, color='#2E86AB')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Best Cost', fontsize=11)
    ax.set_title('Solution Convergence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add improvement annotation
    improvement = cost_history[0] - cost_history[-1]
    improvement_pct = (improvement / cost_history[0]) * 100
    ax.text(0.98, 0.98, f'Improvement: {improvement:.1f} ({improvement_pct:.1f}%)',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Destroy operators
    ax = axes[0, 1]
    colors = plt.cm.Set3(np.linspace(0, 1, len(destroy_ops)))
    bars = ax.bar(destroy_ops, destroy_counts, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Usage Count', fontsize=11)
    ax.set_title('Destroy Operators', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    # Add percentage labels on bars
    total = sum(destroy_counts)
    for bar, count in zip(bars, destroy_counts):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}\n({count/total*100:.1f}%)',
                   ha='center', va='bottom', fontsize=9)
    
    # 3. Removal buckets
    ax = axes[1, 0]
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(buckets)))
    bars = ax.bar(buckets, bucket_counts, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Usage Count', fontsize=11)
    ax.set_xlabel('Removal Size', fontsize=11)
    ax.set_title('Destruction Buckets', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    # Add percentage labels
    for bar, count in zip(bars, bucket_counts):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}\n({count/total*100:.1f}%)',
                   ha='center', va='bottom', fontsize=9)
    
    # 4. Repair operators
    ax = axes[1, 1]
    colors = ['#A8DADC', '#457B9D']
    bars = ax.bar(repair_ops, repair_counts, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Usage Count', fontsize=11)
    ax.set_title('Repair Operators', fontsize=12, fontweight='bold')
    
    # Add percentage labels
    for bar, count in zip(bars, repair_counts):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}\n({count/total*100:.1f}%)',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_path = f'operator_analysis_{instance_name.replace(".txt", "").replace(".TXT", "")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    import glob
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    NUM_RUNS = 10
    ALGORITHM = "DRLH"
    MODEL = os.path.join(project_root, 'logs', 'drl_sb3_20260220_113616', 'ppo_vrptw_final.zip')

    INSTANCE_DIR = os.path.join(project_root, 'Benchmark', 'data', 'homberger_600_1')

    # Find all .TXT and .txt files in the chosen directory
    INSTANCES = sorted(glob.glob(os.path.join(INSTANCE_DIR, '*.TXT')) +
                      glob.glob(os.path.join(INSTANCE_DIR, '*.txt')))


    # Get the last part of the instance directory (e.g., 'homberger_600')
    instance_dir_name = os.path.basename(INSTANCE_DIR.rstrip('/'))
    output_csv = os.path.join(project_root, 'logs', f'drlh_results_{instance_dir_name}_{MAX_ITERATIONS}iter_missing.csv')
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['algorithm', 'instance', 'run', 'best_cost', 'iteration_found_best'])

        for instance_path in INSTANCES:
            instance_name = os.path.basename(instance_path)
            print(f"\n{'='*60}")
            print(f"INSTANCE: {instance_name}")
            print(f"{'='*60}")

            for run in range(1, NUM_RUNS + 1):
                print(f"\n--- Run {run}/{NUM_RUNS} ---")
                start = time.perf_counter()
                cost, _, best_iter = solve_with_sb3(instance_path, MODEL)
                elapsed = time.perf_counter() - start
                print(f"Elapsed time: {elapsed:.3f}s")
                writer.writerow([ALGORITHM, instance_name, run, f"{cost:.2f}", best_iter])
                f.flush()

    print(f"\nResults saved to: {output_csv}")
