import numpy as np

class PDController:
    def __init__(self, Kp, Kd):
        """
        Proportional-Derivative Controller.
        
        Args:
            Kp: Proportional gain matrix (3x3)
            Kd: Derivative gain matrix (3x3)
        """
        self.Kp = Kp
        self.Kd = Kd
        
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
        
        u = -self.Kp @ e_pos - self.Kd @ e_vel
        return u

if __name__ == "__main__":
    # Example usage
    Kp = np.eye(3) * 0.01
    Kd = np.eye(3) * 0.1
    
    controller = PDController(Kp, Kd)
    
    error = np.array([0.001, 0, 0, 0, 0.0001, 0])
    u = controller.compute_control(error)
    
    print(f"Control: {u}")
