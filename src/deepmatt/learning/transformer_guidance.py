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
Transformer-Based Guidance for Manifold Targeting.

This module implements the Deep Manifold Attention architecture with:
- Floquet-Aware Positional Encoding synchronized with orbit period
- Physics-Informed Training with Jacobi integral conservation
- Production-ready training with early stopping and checkpointing

Reference: Paper Section V (Deep RL) and Table 1 (Hyperparameters)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from scipy.integrate import solve_ivp

from deepmatt.dynamics.cr3bp import cr3bp_dynamics
from deepmatt.core.params import L1_HALO_INITIAL_STATE, mu


def jacobi_integral(state, mu_val=mu):
    """
    Compute the Jacobi integral (constant of motion in CR3BP).

    Args:
        state: (N, 6) or (6,) array of [x, y, z, vx, vy, vz]
        mu_val: Mass parameter of the CR3BP system

    Returns:
        C: Jacobi constant value(s)
    """
    state = np.atleast_2d(state)
    x, y, z = state[:, 0], state[:, 1], state[:, 2]
    vx, vy, vz = state[:, 3], state[:, 4], state[:, 5]

    r1 = np.sqrt((x + mu_val) ** 2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu_val) ** 2 + y**2 + z**2)

    Omega = 0.5 * (x**2 + y**2) + (1 - mu_val) / r1 + mu_val / r2
    v_sq = vx**2 + vy**2 + vz**2
    C = 2 * Omega - v_sq

    return C.squeeze()


def jacobi_integral_torch(state, mu_val=mu):
    """
    PyTorch version of Jacobi integral for differentiable loss.

    Args:
        state: (batch, 6) tensor of [x, y, z, vx, vy, vz]
        mu_val: Mass parameter

    Returns:
        C: (batch,) tensor of Jacobi constants
    """
    x, y, z = state[:, 0], state[:, 1], state[:, 2]
    vx, vy, vz = state[:, 3], state[:, 4], state[:, 5]

    r1 = torch.sqrt((x + mu_val) ** 2 + y**2 + z**2 + 1e-10)
    r2 = torch.sqrt((x - 1 + mu_val) ** 2 + y**2 + z**2 + 1e-10)

    Omega = 0.5 * (x**2 + y**2) + (1 - mu_val) / r1 + mu_val / r2
    v_sq = vx**2 + vy**2 + vz**2

    C = 2 * Omega - v_sq
    return C


class FloquetPositionalEncoding(nn.Module):
    """
    Orbit-synchronized positional encoding that injects the natural period
    of the Halo orbit as a prior into the Transformer.

    PE(t) = [sin(2*pi*t/T), cos(2*pi*t/T), sin(4*pi*t/T), cos(4*pi*t/T), ...]

    This is the key contribution enabling manifold awareness.
    """

    def __init__(self, model_dim, orbit_period=3.0, num_harmonics=None):
        """
        Args:
            model_dim: Dimension of the model (must be even)
            orbit_period: Period of the reference Halo orbit (non-dimensional)
            num_harmonics: Number of harmonic frequencies (default: model_dim // 2)
        """
        super().__init__()
        self.model_dim = model_dim
        self.orbit_period = orbit_period
        self.num_harmonics = num_harmonics or model_dim // 2

        # Precompute frequencies
        freqs = torch.tensor(
            [(k + 1) * 2 * np.pi / orbit_period for k in range(self.num_harmonics)]
        )
        self.register_buffer("freqs", freqs)

    def forward(self, timestamps):
        """
        Generate positional encoding for given timestamps.

        Args:
            timestamps: (batch, seq_len) tensor of non-dimensional times

        Returns:
            pe: (batch, seq_len, model_dim) positional encoding
        """
        batch_size, seq_len = timestamps.shape
        device = timestamps.device

        pe = torch.zeros(batch_size, seq_len, self.model_dim, device=device)

        # Vectorized computation
        t_expanded = timestamps.unsqueeze(-1)  # (batch, seq, 1)
        angles = t_expanded * self.freqs  # (batch, seq, num_harmonics)

        for k in range(self.num_harmonics):
            pe[:, :, 2 * k] = torch.sin(angles[:, :, k])
            if 2 * k + 1 < self.model_dim:
                pe[:, :, 2 * k + 1] = torch.cos(angles[:, :, k])

        return pe


class ManifoldTransformer(nn.Module):
    """
    Transformer-based guidance network for predicting optimal manifold
    injection points from noisy state observations.

    Architecture (matches paper Table 1):
    - Model dimension: 64
    - Attention heads: 4
    - Encoder layers: 2
    - Sequence length: 20
    """

    def __init__(
        self,
        input_dim=6,
        model_dim=64,
        num_heads=4,
        num_layers=2,
        output_dim=6,
        orbit_period=3.0,
        dropout=0.1,
    ):
        super(ManifoldTransformer, self).__init__()

        self.model_dim = model_dim
        self.orbit_period = orbit_period

        # Input embedding with layer norm
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        # Floquet-Aware Positional Encoding
        self.pos_encoder = FloquetPositionalEncoding(model_dim, orbit_period)

        # Transformer Encoder with dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output decoder with residual
        self.decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, output_dim),
        )

    def forward(self, x, timestamps=None):
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_dim) state sequences
            timestamps: (batch, seq_len) corresponding timestamps (optional)

        Returns:
            output: (batch, output_dim) predicted target state
        """
        batch_size, seq_len, _ = x.size()

        # Embed input states
        x = self.embedding(x)

        # Add Floquet-aware positional encoding
        if timestamps is not None:
            pe = self.pos_encoder(timestamps)
            x = x + pe
        else:
            # Default timestamps if not provided (for inference)
            default_t = torch.arange(seq_len, device=x.device).float().unsqueeze(0)
            default_t = default_t.expand(batch_size, -1) * 0.01
            pe = self.pos_encoder(default_t)
            x = x + pe

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Predict from last token
        last_token = x[:, -1, :]
        output = self.decoder(last_token)

        return output


class ManifoldDataset(Dataset):
    """
    Dataset of Halo orbit state sequences with timestamps for Floquet PE.
    """

    def __init__(self, seq_len=20, num_samples=10000, dt=0.01, noise_std=1e-4):
        self.seq_len = seq_len
        self.dt = dt
        self.noise_std = noise_std

        # Generate data from Halo orbit
        print(f"  Generating {num_samples} training samples...")
        total_time = (num_samples + seq_len + 10) * dt  # Buffer

        sol = solve_ivp(
            cr3bp_dynamics,
            (0, total_time),
            L1_HALO_INITIAL_STATE,
            method="DOP853",
            dense_output=True,
            rtol=1e-12,
            atol=1e-14,
        )

        if not sol.success:
            raise RuntimeError(f"Orbit integration failed: {sol.message}")

        # Sample at regular intervals
        t_samples = np.arange(0, total_time - seq_len * dt, dt)[:num_samples]

        self.data = []
        self.timestamps = []
        self.targets = []

        for i, t0 in enumerate(t_samples):
            # Input sequence timestamps
            times = t0 + np.arange(seq_len) * dt
            seq = np.array([sol.sol(t) for t in times])

            # Target: next state
            target = sol.sol(t0 + seq_len * dt)

            # Add noise to input (not target)
            noise = np.random.normal(0, noise_std, seq.shape)

            self.data.append(seq + noise)
            self.timestamps.append(times)
            self.targets.append(target)

        print(f"  Dataset ready: {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.data[idx]),
            torch.FloatTensor(self.timestamps[idx]),
            torch.FloatTensor(self.targets[idx]),
        )


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=10, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_transformer(
    lambda_phys=0.1,
    epochs=50,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-5,
    num_samples=10000,
    seq_len=20,
    val_split=0.1,
    patience=10,
    device="auto",
    seed=42,
):
    """
    Train the ManifoldTransformer with physics-informed loss.

    Args:
        lambda_phys: Weight for physics loss (Jacobi conservation)
        epochs: Maximum training epochs
        batch_size: Batch size
        lr: Initial learning rate
        weight_decay: L2 regularization
        num_samples: Number of training samples
        seq_len: Sequence length for input
        val_split: Validation split ratio
        patience: Early stopping patience
        device: Training device (cpu/cuda/auto)
        seed: Random seed
    """
    print("=" * 70)
    print("Deep Manifold Attention - Physics-Informed Training")
    print("=" * 70)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Create dataset
    print("\n[1/4] Generating Manifold Dataset...")
    dataset = ManifoldDataset(seq_len=seq_len, num_samples=num_samples)

    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"  Train samples: {train_size}")
    print(f"  Val samples: {val_size}")

    # Initialize model
    print("\n[2/4] Initializing ManifoldTransformer...")
    model = ManifoldTransformer(input_dim=6, model_dim=64, num_heads=4, num_layers=2)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Loss and optimizer
    criterion_task = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    early_stopping = EarlyStopping(patience=patience)

    print(f"\n[3/4] Training (lambda_physics = {lambda_phys})...")
    print("-" * 70)
    print(
        f"{'Epoch':>5} | {'Train Loss':>12} | {'Val Loss':>12} | {'Task':>10} | {'Physics':>10} | {'LR':>10}"
    )
    print("-" * 70)

    best_val_loss = float("inf")
    os.makedirs("models", exist_ok=True)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_task = 0
        train_phys = 0

        for inputs, timestamps, targets in train_loader:
            inputs = inputs.to(device)
            timestamps = timestamps.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs, timestamps)

            # Task loss
            loss_task = criterion_task(outputs, targets)

            # Physics loss (Jacobi conservation)
            C_pred = jacobi_integral_torch(outputs)
            C_target = jacobi_integral_torch(targets)
            loss_physics = torch.mean((C_pred - C_target) ** 2)

            # Total loss
            loss = loss_task + lambda_phys * loss_physics

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_task += loss_task.item()
            train_phys += loss_physics.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, timestamps, targets in val_loader:
                inputs = inputs.to(device)
                timestamps = timestamps.to(device)
                targets = targets.to(device)

                outputs = model(inputs, timestamps)
                loss_task = criterion_task(outputs, targets)
                C_pred = jacobi_integral_torch(outputs)
                C_target = jacobi_integral_torch(targets)
                loss_physics = torch.mean((C_pred - C_target) ** 2)
                val_loss += (loss_task + lambda_phys * loss_physics).item()

        # Average losses
        n_train = len(train_loader)
        n_val = len(val_loader)
        train_loss /= n_train
        train_task /= n_train
        train_phys /= n_train
        val_loss /= n_val

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch+1:5d} | {train_loss:12.6f} | {val_loss:12.6f} | "
            f"{train_task:10.6f} | {train_phys:10.6f} | {current_lr:10.2e}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/transformer_guidance_best.pth")

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.should_stop:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

        scheduler.step()

    print("-" * 70)
    print("\n[4/4] Saving Model...")

    # Save final model
    torch.save(model.state_dict(), "models/transformer_guidance.pth")

    # Load best model for final save
    model.load_state_dict(torch.load("models/transformer_guidance_best.pth"))
    torch.save(model.state_dict(), "models/transformer_guidance.pth")

    print("  Best model saved to models/transformer_guidance.pth")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print("=" * 70)
    print("Training Complete.")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer guidance network")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--samples", type=int, default=10000, help="Training samples")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--lambda-phys", type=float, default=0.1, help="Physics loss weight"
    )
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_transformer(
        epochs=args.epochs,
        num_samples=args.samples,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_phys=args.lambda_phys,
        device=args.device,
        seed=args.seed,
    )
