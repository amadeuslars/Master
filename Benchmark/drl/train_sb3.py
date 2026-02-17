import sys
import os
import numpy as np
import glob
import time
import random
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from Benchmark.drl.env import VRPTWEnv
from Benchmark.utils.utils import load_raw_solomon_data


class TrainingCallback(BaseCallback):
    """Logging and checkpointing for vectorized environments."""
    
    def __init__(self, save_freq=50000, save_path=None, verbose=0):
        super().__init__(verbose)
        self.start_time = time.time()
        self.save_freq = save_freq
        self.save_path = save_path
        self.last_save_step = 0
        self.last_log_step = 0
    
    def _on_step(self) -> bool:
        # Log metrics every 100 steps
        if self.num_timesteps - self.last_log_step >= 100:
            elapsed = time.time() - self.start_time
            fps = self.num_timesteps / elapsed
            self.logger.record("time/fps", fps)
            self.last_log_step = self.num_timesteps
        
        # Save checkpoint
        if self.num_timesteps - self.last_save_step >= self.save_freq:
            checkpoint_path = os.path.join(self.save_path, f'checkpoint_{self.num_timesteps}')
            self.model.save(checkpoint_path)
            print(f"\n[Checkpoint] Saved at {self.num_timesteps} timesteps\n")
            self.last_save_step = self.num_timesteps
        
        return True


def train_sb3():
    # --- LOGGING SETUP ---
    run_name = f"drl_sb3_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(project_root, 'logs', run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"--- Training with Stable-Baselines3 PPO (Single Env) ---")
    print(f"--- Note: Single env is more stable than DummyVecEnv (which causes PPO deadlocks) ---")
    print(f"--- Logs will be saved to: {log_dir} ---")
    print(f"--- To view: tensorboard --logdir={os.path.join(project_root, 'logs')} ---")
    
    # --- DATA LOADING ---
    data_dirs = [
        os.path.join(project_root, 'Benchmark', 'data', 'homberger_100'),
        os.path.join(project_root, 'Benchmark', 'data', 'homberger_200')
    ]
    instance_files = []
    for data_dir in data_dirs:
        instance_files.extend(glob.glob(os.path.join(data_dir, "*.txt")))
        instance_files.extend(glob.glob(os.path.join(data_dir, "*.TXT")))
    
    if not instance_files:
        print(f"\nError: No .txt files found in {', '.join(data_dirs)}")
        return
    
    print(f"--- Pre-loading {len(instance_files)} instances... ---")
    all_instances_data = []
    
    for file_path in instance_files:
        try:
            data = load_raw_solomon_data(file_path)
            all_instances_data.append(data)
        except Exception as e:
            print(f"Skipping {os.path.basename(file_path)}: {e}")
    
    print(f"--- Successfully loaded {len(all_instances_data)} instances ---")
    
    # --- ENVIRONMENT SETUP ---
    # Wrapper to rotate instances
    class MultiInstanceEnv(VRPTWEnv):
        def __init__(self, instance_list):
            self.instance_list = instance_list
            self.reset_count = 0
            # Initialize with first instance
            super().__init__(*instance_list[0])
        
        def reset(self, **kwargs):
            # Pick new instance and re-initialize
            idx = random.randint(0, len(self.instance_list) - 1)
            instance_data = self.instance_list[idx]
            
            # Log first 5 resets to verify rotation
            if self.reset_count < 5:
                num_custs = len(instance_data[0]['customer_id']) if isinstance(instance_data[0], dict) else len(instance_data[0])
                print(f"  [Reset {self.reset_count}] Instance {idx}, {num_custs} customers")
            self.reset_count += 1
            
            # Call parent init with new instance
            VRPTWEnv.__init__(self, *instance_data)
            
            # Now call parent reset
            return VRPTWEnv.reset(self, **kwargs)
    
    env = MultiInstanceEnv(all_instances_data)
    
    print(f"--- Created multi-instance environment ({len(all_instances_data)} instances) ---")
    
    # --- MODEL SETUP ---
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-04,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  
        verbose=1,
        tensorboard_log=log_dir,
        device="cpu"
    )
    
    # --- TRAINING ---
    MAX_TIMESTEPS = 1_000_000
    CHECKPOINT_FREQ = 50_000
    
    print(f"\n--- Starting training for {MAX_TIMESTEPS} timesteps ---\n")
    
    model.learn(
        total_timesteps=MAX_TIMESTEPS,
        callback=TrainingCallback(save_freq=CHECKPOINT_FREQ, save_path=log_dir),
        tb_log_name="PPO_VRPTW"
    )
    
    # --- SAVE MODEL ---
    final_model_path = os.path.join(log_dir, 'ppo_vrptw_final')
    model.save(final_model_path)
    print(f"\nTraining complete. Final model saved to {final_model_path}")


if __name__ == "__main__":
    train_sb3()
