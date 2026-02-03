import torch
import numpy as np
import sys
import os

# --- PATH SETUP ---
# Get the absolute path of the script (.../Benchmark/drl/solve.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to get Project_Root (.../Master/Master)
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from Benchmark.drl.env import VRPTWEnv
from Benchmark.drl.agent import PPOAgent
from Benchmark.utils.utils import load_raw_solomon_data

def solve_with_drl():
    # --- CONFIGURATION ---
    # Must match the instance you trained on (or similar type)
    INSTANCE_NAME = 'r106.txt'  
    
    # Construct absolute path to the instance
    file_path = os.path.join(project_root, 'Benchmark', 'Instances', INSTANCE_NAME)
    model_path = os.path.join(current_dir, 'drlh_vrptw_model.pth')
    
    print(f"--- Loading Raw Data from: {file_path} ---")
    
    if not os.path.exists(file_path):
        print(f"\n Error: Instance file not found at {file_path}")
        return

    # 1. Load Data
    data = load_raw_solomon_data(file_path)
    customers_df, vehicles_df, dist_matrix, cust_addr_idx, cust_arrays = data
    
    # 2. Setup Environment
    env = VRPTWEnv(customers_df, vehicles_df, dist_matrix, cust_addr_idx, cust_arrays)
    
    # 3. Load the Trained Agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim, action_dim)
    
    if os.path.exists(model_path):
        agent.policy.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Loaded trained model from: {model_path}")
    else:
        print(f"Model not found at {model_path}! Running with random weights.")

    # 4. Run Solver (Inference Loop)
    state, _ = env.reset()
    done = False
    print("\nStarting DRL Solver...")
    
    step_count = 0
    while not done:
        step_count += 1
        
        # Get action from Neural Net (Deterministic / Greedy for inference)
        # We grab the actor network directly to get the probabilities
        state_tensor = torch.from_numpy(state).float().to(agent.policy.actor[0].weight.device)
        
        with torch.no_grad():
            probs = agent.policy.actor(state_tensor)
            # Pick the action with the highest probability (Argmax)
            action = torch.argmax(probs).item()
            
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Optional: Print progress every 50 steps
        if step_count % 50 == 0:
            print(f"Step {step_count}: Best Cost = {env.best_sol._cost:.2f}")
        
    # 5. Final Report
    print("\n" + "="*40)
    print(f"FINAL RESULT ({INSTANCE_NAME})")
    print("="*40)
    print(f"Best Cost Found: {env.best_sol._cost:.2f}")
    
    print("\nRoutes:")
    vehicle_count = 0
    for i, r in enumerate(env.best_sol.routes[:-1]):
        if r:
            vehicle_count += 1
            load = sum(cust_arrays['demand'][c-1] for c in r)
            print(f"Vehicle {i+1}: {r} | Load: {load}")
            
    print(f"\nTotal Vehicles Used: {vehicle_count}")

if __name__ == "__main__":
    solve_with_drl()