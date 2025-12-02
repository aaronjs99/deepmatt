import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from cislune.dynamics.cr3bp import cr3bp_dynamics
from cislune.core.params import L1_HALO_INITIAL_STATE

class ManifoldTransformer(nn.Module):
    def __init__(self, input_dim=6, model_dim=64, num_heads=4, num_layers=2, output_dim=6):
        super(ManifoldTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, model_dim)) # Max seq len 100
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.decoder = nn.Linear(model_dim, output_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        x = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        x = self.transformer_encoder(x)
        
        # Predict based on the last token's embedding
        last_token = x[:, -1, :]
        output = self.decoder(last_token)
        return output

class ManifoldDataset(Dataset):
    def __init__(self, seq_len=10, num_samples=1000):
        self.seq_len = seq_len
        self.data = []
        self.targets = []
        
        # Generate data from Halo orbit
        # Propagate for a few periods
        t_span = np.linspace(0, 20, num_samples + seq_len + 1)
        from scipy.integrate import solve_ivp
        sol = solve_ivp(cr3bp_dynamics, (0, 20), L1_HALO_INITIAL_STATE, t_eval=t_span, rtol=1e-9, atol=1e-12)
        states = sol.y.T # (N, 6)
        
        for i in range(num_samples):
            # Input: Sequence of states
            seq = states[i : i+seq_len]
            # Target: Next state (prediction)
            target = states[i+seq_len]
            
            # Add noise to input to make it robust
            noise = np.random.normal(0, 1e-4, seq.shape)
            self.data.append(seq + noise)
            self.targets.append(target)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.FloatTensor(self.targets[idx])

def train_transformer():
    print("Generating Manifold Dataset...")
    dataset = ManifoldDataset(seq_len=20, num_samples=5000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = ManifoldTransformer()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting Training...")
    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")
        
    print("Saving Model...")
    torch.save(model.state_dict(), "models/transformer_guidance.pth")
    print("Training Complete.")

if __name__ == "__main__":
    train_transformer()
