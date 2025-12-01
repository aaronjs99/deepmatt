import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

from cislune.core.params import mu, L1_HALO_INITIAL_STATE
from cislune.dynamics.cr3bp import cr3bp_dynamics as halo_dynamics, linearized_dynamics
from cislune.gnc.guidance import find_natural_formation
from cislune.gnc.navigation import EKF
from cislune.gnc.actuators import Thruster

def integrated_gnc_sim():
    print("Running Integrated GNC Simulation (Guidance, Navigation, Control)...")
    
    # 1. Guidance: Find Natural Formation
    chief_initial = np.array(L1_HALO_INITIAL_STATE)
    period = 3.0
    
    print("Guidance: Calculating natural formation offsets...")
    natural_dirs = find_natural_formation(chief_initial, period)
    if len(natural_dirs) == 0:
        print("No natural directions found. Using arbitrary offset.")
        target_rel_state = np.array([1e-4, 0, 0, 0, 0, 0])
    else:
        # Pick one (e.g., the first one, scaled)
        # Scale to ~10 km (10/384400 ~ 2.6e-5)
        scale = 2.6e-5
        target_rel_state = natural_dirs[0] * scale
        print(f"Target Relative State (Natural): {target_rel_state}")

    # 2. Initialization
    # True States
    x_chief_true = chief_initial.copy()
    x_deputy_true = chief_initial + target_rel_state
    
    # Navigation Initialization (EKF)
    # Initial estimate error
    est_error = np.random.normal(0, 1e-6, 6)
    x_rel_est = target_rel_state + est_error
    
    P0 = np.eye(6) * 1e-10
    Q = np.eye(6) * 1e-12 # Process noise (model mismatch)
    R = np.eye(3) * 1e-10 # Measurement noise (sensor)
    
    ekf = EKF(x_rel_est, P0, Q, R)
    
    # Actuator Initialization
    thruster = Thruster(noise_std=1e-5, bias=1e-6)
    
    # Control Gains (LQR)
    Q_lqr = np.diag([1e4, 1e4, 1e4, 1e2, 1e2, 1e2])
    R_lqr = np.eye(3) * 1e-1
    
    def get_gain(x):
        A = linearized_dynamics(x)
        B = np.zeros((6, 3))
        B[3:, :] = np.eye(3)
        try:
            P = solve_continuous_are(A, B, Q_lqr, R_lqr)
            K = np.linalg.inv(R_lqr) @ B.T @ P
            return K
        except:
            return np.zeros((3, 6))
            
    # Simulation Loop
    dt = 0.01
    t_max = 3.0
    steps = int(t_max / dt)
    time = np.linspace(0, t_max, steps)
    
    history = {
        't': [],
        'pos_err_true': [],
        'est_err': [],
        'control': [],
        'fuel': 0.0
    }
    
    print("Starting Simulation Loop...")
    
    for t in time:
        # A. Navigation Update
        # Simulate Sensor Measurement (True Relative Pos + Noise)
        rel_pos_true = x_deputy_true[:3] - x_chief_true[:3]
        meas_noise = np.random.normal(0, 1e-5, 3) # ~4 meters noise
        z = rel_pos_true + meas_noise
        
        # EKF Update
        ekf.update(z)
        x_est = ekf.x # Estimated relative state
        
        # B. Guidance & Control
        # Target: Maintain the natural relative orbit?
        # If it's a natural orbit, x_target(t) evolves naturally.
        # But here we want to regulate TO the natural orbit if we drift.
        # Error = Estimated - Target
        # Wait, if we start ON the natural orbit, target is just current state?
        # No, we want to stay on the reference natural trajectory.
        # Let's propagate the target relative state analytically or numerically.
        # For simplicity, let's regulate to ZERO deviation from the initial natural offset propagated?
        # Actually, let's just regulate to the *initial* relative state if it's a fixed point (it's not).
        # Let's assume we want to track the *linearized* natural trajectory.
        # x_target_k+1 = Phi * x_target_k
        
        # Propagate target (ideal)
        # For this step, we just use the current estimate as error? No.
        # We need a reference trajectory.
        # Let's generate the reference trajectory beforehand or propagate it.
        if t == 0:
            x_ref_rel = target_rel_state
        else:
            # Propagate reference using STM
            A_ref = linearized_dynamics(x_chief_true)
            x_ref_rel = x_ref_rel + (A_ref @ x_ref_rel) * dt
            
        # Control Error
        err_state = x_est - x_ref_rel
        
        # LQR Control
        K = get_gain(x_chief_true)
        u_cmd = -K @ err_state
        
        # C. Actuation
        u_act = thruster.fire(u_cmd)
        history['fuel'] += np.linalg.norm(u_act) * dt
        
        # D. Propagation (Physics)
        # Propagate Chief (Natural)
        sol_c = solve_ivp(halo_dynamics, (0, dt), x_chief_true, method="RK45")
        x_chief_next = sol_c.y[:, -1]
        
        # Propagate Deputy (Natural + Control)
        def dep_dyn(t, x):
            d = halo_dynamics(t, x)
            d[3:] += u_act
            return d
            
        sol_d = solve_ivp(dep_dyn, (0, dt), x_deputy_true, method="RK45")
        x_deputy_next = sol_d.y[:, -1]
        
        # E. Navigation Prediction
        ekf.predict(u_cmd, dt, x_chief_true)
        
        # Store Data
        history['t'].append(t)
        history['pos_err_true'].append(np.linalg.norm((x_deputy_true - x_chief_true) - x_ref_rel)) # Deviation from reference
        history['est_err'].append(np.linalg.norm(x_est - (x_deputy_true - x_chief_true))) # Estimation error
        history['control'].append(np.linalg.norm(u_act))
        
        # Update
        x_chief_true = x_chief_next
        x_deputy_true = x_deputy_next
        
    # Analysis
    V_star = 1024.0
    total_dv = history['fuel'] * V_star
    
    print("\n--- GNC Simulation Results ---")
    print(f"Total Delta-V: {total_dv:.2f} m/s")
    print(f"Final Position Error (True vs Ref): {history['pos_err_true'][-1] * 384400 * 1000:.2f} meters")
    print(f"Final Estimation Error: {history['est_err'][-1] * 384400 * 1000:.2f} meters")
    
    # Plot
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    
    ax[0].plot(history['t'], np.array(history['pos_err_true']) * 384400, label='True Deviation (km)')
    ax[0].set_title('Formation Keeping Error')
    ax[0].set_ylabel('Error (km)')
    ax[0].legend()
    
    ax[1].plot(history['t'], np.array(history['est_err']) * 384400 * 1000, label='Estimation Error (m)', color='orange')
    ax[1].set_title('Navigation Performance (EKF)')
    ax[1].set_ylabel('Error (m)')
    ax[1].legend()
    
    ax[2].plot(history['t'], history['control'], label='Control Effort', color='green')
    ax[2].set_title('Control Usage')
    ax[2].set_ylabel('Acceleration')
    ax[2].set_xlabel('Time')
    
    plt.tight_layout()
    plt.savefig("results/formation/gnc_performance.png")
    print("Saved plot to results/formation/gnc_performance.png")

if __name__ == "__main__":
    integrated_gnc_sim()
