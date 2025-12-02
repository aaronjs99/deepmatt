import time
import numpy as np
import torch
from stable_baselines3 import PPO
from cislune.learning.transformer_guidance import ManifoldTransformer
from cislune.gnc.control import SDREController
from cislune.core.params import L1_HALO_INITIAL_STATE

def benchmark_compute():
    print("--- Computational Benchmark ---")
    
    # Setup
    n_trials = 1000
    state = np.array([0.1, 0.0, 0.0, 0.0, 0.01, 0.0]) # Relative state
    
    # 1. SDRE Benchmark
    controller = SDREController()
    # Warmup
    controller.compute_control(state, 0.0)
    
    start_time = time.time()
    for _ in range(n_trials):
        controller.compute_control(state, 0.0)
    end_time = time.time()
    sdre_time = (end_time - start_time) / n_trials * 1000 # ms
    print(f"SDRE Mean Inference Time: {sdre_time:.4f} ms")
    
    # 2. PPO Benchmark
    try:
        ppo_model = PPO.load("models/ppo_formation.zip")
        # Warmup
        ppo_model.predict(state, deterministic=True)
        
        start_time = time.time()
        for _ in range(n_trials):
            ppo_model.predict(state, deterministic=True)
        end_time = time.time()
        ppo_time = (end_time - start_time) / n_trials * 1000 # ms
        print(f"PPO Mean Inference Time: {ppo_time:.4f} ms")
    except:
        print("PPO model not found, skipping benchmark.")
        ppo_time = 0.0

    # 3. Transformer Benchmark
    try:
        transformer = ManifoldTransformer()
        # Mock input (batch=1, seq=20, feat=6)
        inp = torch.randn(1, 20, 6)
        # Warmup
        with torch.no_grad():
            transformer(inp)
            
        start_time = time.time()
        for _ in range(n_trials):
            with torch.no_grad():
                transformer(inp)
        end_time = time.time()
        trans_time = (end_time - start_time) / n_trials * 1000 # ms
        print(f"Transformer Mean Inference Time: {trans_time:.4f} ms")
    except:
        print("Transformer model not found, skipping benchmark.")
        trans_time = 0.0
        
    # Save results to file for paper inclusion
    with open("results/validation/compute_benchmark.txt", "w") as f:
        f.write(f"SDRE: {sdre_time:.4f} ms\n")
        f.write(f"PPO: {ppo_time:.4f} ms\n")
        f.write(f"Transformer: {trans_time:.4f} ms\n")

if __name__ == "__main__":
    benchmark_compute()
