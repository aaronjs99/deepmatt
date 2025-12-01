import numpy as np

class SlidingModeController:
    def __init__(self, lambd, K, epsilon=0.01):
        """
        Sliding Mode Controller.
        
        Args:
            lambd: Sliding surface slope (scalar or 3,)
            K: Switching gain (scalar or 3,)
            epsilon: Boundary layer thickness
        """
        self.lambd = lambd
        self.K = K
        self.epsilon = epsilon
        
    def compute_control(self, error_state):
        """
        Compute control input.
        
        Args:
            error_state: [position_error(3), velocity_error(3)]
        
        Returns:
            u: Control acceleration (3,)
        """
        e_pos = error_state[:3]
        e_vel = error_state[3:]
        
        # Sliding surface: s = e_vel + lambda * e_pos
        s = e_vel + self.lambd * e_pos
        
        # Sat function (smooth approximation of sign)
        def sat(x, eps):
            return np.clip(x / eps, -1, 1)
        
        # Control: u = -K * sat(s)
        u = -self.K * sat(s, self.epsilon)
        
        return u

if __name__ == "__main__":
    # Example usage
    controller = SlidingModeController(lambd=2.0, K=0.1)
    
    error = np.array([0.001, 0, 0, 0, 0.0001, 0])
    u = controller.compute_control(error)
    
    print(f"Control: {u}")
