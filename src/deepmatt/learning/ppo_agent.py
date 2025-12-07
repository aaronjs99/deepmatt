# /***********************************************************
# *                                                         *
# * Copyright (c) 2025                                      *
# *                                                         *
# * University of California, Los Angeles                   *
# *                                                         *
# * Author : Aaron John-Sabu                                *
# * Contact: aaronjs@g.ucla.edu                             *
# *                                                         *
# ***********************************************************/

"""
PPO Agent for Formation Control in Cislunar Space.

This module provides production-ready PPO training with:
- Configurable hyperparameters matching paper specifications
- Callback-based logging and checkpointing
- Proper evaluation during training
- Robust error handling
"""

import numpy as np
import os
import argparse
from scipy.integrate import solve_ivp

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from deepmatt.core.params import L1_HALO_INITIAL_STATE
from deepmatt.dynamics.cr3bp import cr3bp_dynamics
from deepmatt.learning.env import DeepMAttFormationEnv


def generate_chief_orbit(period=3.0, dt=0.01, method="DOP853"):
    """
    Generate the reference Halo orbit for the environment.

    Args:
        period: Number of orbital periods to simulate (non-dimensional)
        dt: Time step (non-dimensional, ~1 hour physical)
        method: Integration method (RK45, DOP853, etc.)

    Returns:
        states: (N, 6) array of chief spacecraft states
    """
    t_eval = np.arange(0, period, dt)
    sol = solve_ivp(
        cr3bp_dynamics,
        (0, period),
        L1_HALO_INITIAL_STATE,
        method=method,
        t_eval=t_eval,
        rtol=1e-12,
        atol=1e-14,
    )

    if not sol.success:
        raise RuntimeError(f"Orbit integration failed: {sol.message}")

    return sol.y.T


def make_env(chief_history, max_steps=1000, seed=None):
    """Create and wrap the environment with monitoring."""

    def _init():
        env = DeepMAttFormationEnv(chief_history, max_steps=max_steps)
        if seed is not None:
            env.reset(seed=seed)
        return Monitor(env)

    return _init


def train_ppo(
    total_timesteps=500_000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs=None,
    save_freq=50_000,
    eval_freq=10_000,
    n_eval_episodes=10,
    seed=42,
    device="auto",
    verbose=1,
):
    """
    Train PPO agent for formation control.

    Hyperparameters match the paper specification (Table 1):
    - Policy Network: MLP (64, 64) with Tanh activation
    - Learning rate: 3e-4
    - Discount factor: 0.99
    - GAE lambda: 0.95
    - Clip range: 0.2

    Args:
        total_timesteps: Total training timesteps
        learning_rate: Adam optimizer learning rate
        n_steps: Steps per rollout
        batch_size: Minibatch size
        n_epochs: Epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda for advantage estimation
        clip_range: PPO clipping parameter
        ent_coef: Entropy coefficient for exploration
        policy_kwargs: Custom policy architecture
        save_freq: Checkpoint save frequency
        eval_freq: Evaluation frequency
        n_eval_episodes: Episodes per evaluation
        seed: Random seed
        device: Training device (cpu/cuda/auto)
        verbose: Verbosity level
    """
    print("=" * 70)
    print("DeepMAtt PPO Training - Production Configuration")
    print("=" * 70)

    # Set seeds for reproducibility
    np.random.seed(seed)

    # Default policy architecture (matches paper Table 1)
    if policy_kwargs is None:
        policy_kwargs = dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=torch.nn.Tanh,
        )

    # Generate reference orbit
    print("\n[1/5] Generating reference Halo orbit...")
    chief_history = generate_chief_orbit(period=3.0, dt=0.01)
    print(
        f"      Generated {len(chief_history)} states ({len(chief_history)*0.01:.1f} T)"
    )

    # Create environments
    print("[2/5] Initializing environments...")
    train_env = DummyVecEnv([make_env(chief_history, seed=seed)])
    eval_env = DummyVecEnv([make_env(chief_history, seed=seed + 1000)])

    # Validate environment
    check_env(DeepMAttFormationEnv(chief_history), warn=True)

    # Setup directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Callbacks
    print("[3/5] Setting up callbacks...")
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="models/",
        name_prefix="ppo_formation",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="logs/",
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # Initialize PPO
    print("[4/5] Initializing PPO agent...")
    print(f"      Policy: MLP(64, 64) + Tanh")
    print(f"      Learning rate: {learning_rate}")
    print(f"      Total timesteps: {total_timesteps:,}")

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./ppo_deepmatt_tensorboard/",
        verbose=verbose,
        seed=seed,
        device=device,
    )

    # Training
    print("[5/5] Training...")
    print("-" * 70)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")

    # Save final model
    final_path = "models/ppo_formation_final"
    model.save(final_path)
    print("-" * 70)
    print(f"✓ Final model saved to {final_path}.zip")
    print("=" * 70)
    print("Training Complete.")

    return model


if __name__ == "__main__":
    import torch

    parser = argparse.ArgumentParser(
        description="Train PPO agent for formation control"
    )
    parser.add_argument(
        "--timesteps", type=int, default=500_000, help="Total training timesteps"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (cpu/cuda/auto)"
    )
    args = parser.parse_args()

    train_ppo(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        seed=args.seed,
        device=args.device,
    )
