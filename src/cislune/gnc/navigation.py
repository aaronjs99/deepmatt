import numpy as np
from cislune.dynamics.cr3bp import linearized_dynamics

class EKF:
    def __init__(self, x0, P0, Q, R):
        """
        Extended Kalman Filter for Relative Navigation.
        State: Relative Position and Velocity (6x1)
        """
        self.x = x0 # Estimate
        self.P = P0 # Covariance
        self.Q = Q  # Process Noise Covariance
        self.R = R  # Measurement Noise Covariance
        
    def predict(self, u, dt, chief_state):
        """
        Propagate state and covariance.
        Model: x_k+1 = Phi * x_k + B * u
        """
        # Linearize dynamics around Chief (or current estimate + Chief)
        # For relative motion, A is evaluated at Chief's state
        A = linearized_dynamics(chief_state)
        
        # Discretize (First order Euler for covariance, or expm)
        # Phi = I + A*dt
        Phi = np.eye(6) + A * dt
        
        # Propagate State (Linearized relative dynamics)
        # dx_dot = A * dx + B * u
        # x_new = x + (A*x + B*u)*dt
        # B matrix maps control (accel) to state derivative
        B = np.zeros((6, 3))
        B[3:, :] = np.eye(3)
        
        dx = A @ self.x + B @ u
        self.x = self.x + dx * dt
        
        # Propagate Covariance
        # P = Phi * P * Phi.T + Q
        self.P = Phi @ self.P @ Phi.T + self.Q
        
    def update(self, z):
        """
        Update with measurement.
        Measurement: Relative Position (3x1) (assuming GPS/Link gives pos directly for simplicity, or Range/Bearing converted)
        Let's assume we measure [x, y, z] directly with noise.
        H = [I 0]
        """
        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        
        # Innovation
        y = z - H @ self.x
        
        # Innovation Covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman Gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update State
        self.x = self.x + K @ y
        
        # Update Covariance
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P
        
        return self.x
