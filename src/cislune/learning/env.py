import gymnasium as gym
from gymnasium import spaces
import numpy as np
from cislune.dynamics.cr3bp import cr3bp_dynamics
from cislune.core.params import mu, L_star, T_star

class CisluneFormationEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This environment simulates the relative motion of a deputy spacecraft
    near a chief in a Halo orbit.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, chief_history, max_steps=1000):
        super(CisluneFormationEnv, self).__init__()
        
        self.chief_history = chief_history # (N, 6) array of chief state history
        self.max_steps = max_steps
        self.current_step = 0
        self.dt = 0.01 # Non-dimensional time step (approx 1 hour)
        
        # Action space: Continuous thrust command [ux, uy, uz]
        # Normalized to [-1, 1], scaled by max thrust in step
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation space: Relative state [x, y, z, vx, vy, vz]
        # We assume relative position is within +/- 100 km and velocity +/- 10 m/s
        # Normalized values roughly
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.state = None
        self.chief_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomize initial relative state slightly
        # Position error ~ 1 km, Velocity error ~ 1 cm/s
        pos_err = np.random.normal(0, 1e-5, 3) 
        vel_err = np.random.normal(0, 1e-6, 3)
        self.state = np.concatenate([pos_err, vel_err])
        
        # Start at a random point in the chief's orbit (or 0)
        self.chief_idx = 0 
        self.current_step = 0
        
        return self.state.astype(np.float32), {}

    def step(self, action):
        # Apply scaling to action (max thrust)
        # Assume max acceleration ~ 1e-4 non-dimensional (approx 0.27 mm/s^2)
        u_max = 1e-4
        u_cmd = action * u_max
        
        # Get current chief state
        xc = self.chief_history[self.chief_idx]
        
        # Dynamics propagation (RK4 or simple Euler for RL speed)
        # Using simple Euler for now for speed, can upgrade to RK4
        # Relative dynamics: f(xd) - f(xc)
        
        xd = xc + self.state
        
        # Compute derivatives
        # cr3bp_dynamics returns a list, so we cast to numpy array
        dxd_nat = np.array(cr3bp_dynamics(0, xd)) 
        dxc = np.array(cr3bp_dynamics(0, xc))
        
        # Add control acceleration to deputy
        dxd = dxd_nat.copy()
        dxd[3:] += u_cmd
        
        # Update relative state
        d_rel = dxd - dxc
        self.state += d_rel * self.dt
        
        # Update time/index
        self.current_step += 1
        self.chief_idx = (self.chief_idx + 1) % len(self.chief_history)
        
        # Calculate Reward
        # Minimize position error and fuel
        pos_error = np.linalg.norm(self.state[:3])
        vel_error = np.linalg.norm(self.state[3:])
        fuel_cost = np.linalg.norm(u_cmd)
        
        reward = - (10.0 * pos_error + 1.0 * vel_error + 100.0 * fuel_cost)
        
        # Check termination
        terminated = False
        truncated = False
        
        if pos_error > 0.1: # Diverged > ~38,000 km (too loose, maybe 0.01 for 3800km)
             # Let's say 100km limit -> 100/384400 ~ 2.6e-4
             # But for learning, let's keep it loose initially
             pass
             
        if self.current_step >= self.max_steps:
            truncated = True
            
        return self.state.astype(np.float32), reward, terminated, truncated, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
