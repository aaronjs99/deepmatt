import numpy as np
from scipy.integrate import solve_ivp
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os

from cislune.core.params import L1_HALO_INITIAL_STATE
from cislune.dynamics.cr3bp import cr3bp_dynamics
from cislune.learning.env import CisluneFormationEnv

def generate_chief_orbit(period=3.0, dt=0.01):
    """
    Generate the reference Halo orbit for the environment.
    """
    t_eval = np.arange(0, period, dt)
    sol = solve_ivp(cr3bp_dynamics, (0, period), L1_HALO_INITIAL_STATE, t_eval=t_eval, rtol=1e-9, atol=1e-12)
    return sol.y.T # (N, 6)

def train_ppo():
    print("Generating reference Halo orbit...")
    chief_history = generate_chief_orbit()
    
    print("Initializing Environment...")
    env = CisluneFormationEnv(chief_history)
    
    # Check if env follows Gym API
    check_env(env)
    
    print("Defining PPO Agent...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_cislune_tensorboard/")
    
    print("Starting Training...")
    model.learn(total_timesteps=100000)
    
    print("Saving Model...")
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_formation")
    
    print("Training Complete.")

if __name__ == "__main__":
    train_ppo()
