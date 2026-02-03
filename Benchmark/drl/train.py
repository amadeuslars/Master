import sys
import os
import numpy as np
import torch
import glob
import random
import csv
import time
from torch.utils.tensorboard import SummaryWriter

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from Benchmark.drl.env import VRPTWEnv
from Benchmark.drl.agent import PPOAgent
from Benchmark.utils.utils import load_raw_solomon_data

def train_drlh():
    # --- LOGGING SETUP ---
    # Create unique run name based on time
    run_name = f"drl_run_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(project_root, 'logs', run_name)
    writer = SummaryWriter(log_dir=log_dir)
    
    # CSV file for raw data (backup)
    os.makedirs(log_dir, exist_ok=True)
    csv_file = os.path.join(log_dir, 'training_log.csv')
    with open(csv_file, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Episode', 'Instance', 'Best_Cost', 'Total_Reward', 'Steps', 'Avg_Success_Rate'])

    print(f"--- Logs will be saved to: {log_dir} ---")
    print(f"--- To view: tensorboard --logdir={os.path.join(project_root, 'logs')} ---")

    # --- DATA LOADING ---
    instances_dir = os.path.join(project_root, 'Benchmark', 'Instances', 'Training')
    instance_files = glob.glob(os.path.join(instances_dir, "*.txt"))
    
    if not instance_files:
        print(f"\nError: No .txt files found in {instances_dir}")
        return

    print(f"--- Pre-loading {len(instance_files)} instances... ---")
    all_instances_data = []
    instance_names = [] # Keep track of names for logging
    
    for file_path in instance_files:
        try:
            data = load_raw_solomon_data(file_path)
            all_instances_data.append(data)
            instance_names.append(os.path.basename(file_path))
        except Exception as e:
            print(f"Skipping {os.path.basename(file_path)}: {e}")

    # --- AGENT SETUP ---
    temp_env = VRPTWEnv(*all_instances_data[0])
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.n
    agent = PPOAgent(state_dim, action_dim, lr=0.001, K_epochs=4)
    
    # Define Action Names for Logging
    action_names = [f"{d.__name__}_{r.__name__}" for d, r in temp_env.action_pairs]

    # --- TRAINING LOOP ---
    MAX_EPISODES = 5000  
    UPDATE_TIMESTEP = 200 
    CHECKPOINT_FREQ = 500
    
    time_step = 0
    
    for i_episode in range(1, MAX_EPISODES + 1):
        # Pick random instance
        idx = random.randint(0, len(all_instances_data) - 1)
        selected_data = all_instances_data[idx]
        current_instance_name = instance_names[idx]
        
        env = VRPTWEnv(*selected_data)
        state, _ = env.reset()
        
        current_ep_reward = 0
        done = False
        step_count = 0
        
        while not done:
            time_step += 1
            step_count += 1
            action = agent.select_action(state)
            
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_reward(reward, done)
            current_ep_reward += reward
            
            # PPO Update
            if time_step % UPDATE_TIMESTEP == 0:
                agent.update()
        
        # --- LOGGING PER EPISODE ---
        
        # 1. Scalar Logs (Performance)
        writer.add_scalar('Reward/Total_Reward', current_ep_reward, i_episode)
        writer.add_scalar('Performance/Best_Cost', info['best_cost'], i_episode)
        
        # 2. Operator Usage (The important part for your Thesis)
        # Normalize counts to frequencies to see which op is preferred
        total_actions = info['action_counts'].sum()
        if total_actions > 0:
            usage_freq = info['action_counts'] / total_actions
            # Log as a dictionary of scalars
            for i, freq in enumerate(usage_freq):
                writer.add_scalar(f'Operators_Usage/{action_names[i]}', freq, i_episode)
            
            # Calculate Success Rate
            success_rate = np.divide(info['improvement_counts'], info['action_counts'], 
                                     out=np.zeros_like(info['improvement_counts']), 
                                     where=info['action_counts']!=0)
            avg_success = np.mean(success_rate)
            writer.add_scalar('Operators/Avg_Success_Rate', avg_success, i_episode)
        else:
            avg_success = 0

        # 3. CSV Log
        with open(csv_file, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([i_episode, current_instance_name, info['best_cost'], current_ep_reward, step_count, avg_success])

        # 4. Console Print
        if i_episode % 10 == 0:
            print(f"Ep {i_episode} | Instance: {current_instance_name} | Best: {info['best_cost']:.2f} | Reward: {current_ep_reward:.2f}")

        # 5. Save Model Checkpoint
        if i_episode % CHECKPOINT_FREQ == 0:
            ckpt_path = os.path.join(log_dir, f'drl_checkpoint_{i_episode}.pth')
            torch.save(agent.policy.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Final Save
    final_path = os.path.join(current_dir, 'drlh_vrptw_model.pth')
    torch.save(agent.policy.state_dict(), final_path)
    writer.close()
    print(f"Training Complete. Final model saved to {final_path}")

if __name__ == "__main__":
    train_drlh()