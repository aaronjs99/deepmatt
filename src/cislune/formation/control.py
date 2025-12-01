import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from cislune.core.params import mu, L1_HALO_INITIAL_STATE
from cislune.dynamics.cr3bp import cr3bp_dynamics as halo_dynamics

from scipy.linalg import solve_continuous_are
from cislune.dynamics.cr3bp import linearized_dynamics

def formation_flight():
    print("Simulating Formation Flying in Halo Orbit...")
    
    # Chief initial state (L1 Halo Orbit)
    chief_initial = np.array(L1_HALO_INITIAL_STATE)
    
    # Deputy initial state (Perturbed from Chief)
    # 10 km in dimensionless units: 10 / 384400
    dist_dim = 10.0 / 384400.0
    deputy_initial = chief_initial.copy()
    deputy_initial[0] += dist_dim
    
    t_span = (0, 3.0) # Approx one period
    t_eval = np.linspace(0, 3.0, 1000)
    
    # Integrate Chief (Reference Trajectory)
    sol_chief = solve_ivp(
        halo_dynamics,
        t_span,
        chief_initial,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12
    )
    chief_states = sol_chief.y.T
    
    # --- Uncontrolled Deputy ---
    sol_deputy_unc = solve_ivp(
        halo_dynamics,
        t_span,
        deputy_initial,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12
    )
    deputy_unc_states = sol_deputy_unc.y.T
    
    # --- Controlled Deputy (LQR / SDRE) ---
    print("Simulating Controlled Deputy...")
    
    # Control Parameters
    Q = np.diag([1e4, 1e4, 1e4, 1e1, 1e1, 1e1]) # State penalty
    R = np.eye(3) * 1e-2 # Control effort penalty
    
    def controlled_dynamics(t, y):
        # Interpolate Chief state at time t
        # For simplicity in this demo, we'll just find the closest index or re-integrate
        # Better: pass chief solution as spline
        # Here we use the fact that we are integrating together or just use the dynamics
        
        # Actually, let's integrate relative dynamics directly or use a coupled system
        # For this demo, let's assume we know the Chief's state perfectly (it's the reference)
        
        # To make it robust, let's integrate both simultaneously in the loop if we were doing manual integration
        # But since we use solve_ivp, we need a function that knows the reference.
        # Let's use a wrapper that integrates both 12 states (Chief + Deputy)
        
        # State: [Chief (6), Deputy (6)]
        xc = y[:6]
        xd = y[6:]
        
        # Chief dynamics (Natural)
        dxc = halo_dynamics(t, xc)
        
        # Deputy dynamics (Controlled)
        # u = -K * (xd - xc)
        # Linearize around Chief
        A = linearized_dynamics(xc)
        B = np.zeros((6, 3))
        B[3:, :] = np.eye(3)
        
        # Solve ARE for K
        # A.T P + P A - P B R^-1 B.T P + Q = 0
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        
        error = xd - xc
        u = -K @ error
        
        # Deputy natural dynamics + control
        # Acceleration from gravity
        dxd_nat = halo_dynamics(t, xd)
        
        # Add control acceleration
        dxd = np.array(dxd_nat)
        dxd[3:] += u
        
        return np.concatenate([dxc, dxd])

    # Initial state for coupled system
    y0 = np.concatenate([chief_initial, deputy_initial])
    
    sol_controlled = solve_ivp(
        controlled_dynamics,
        t_span,
        y0,
        method="RK45", # RK45 is usually fine for this, DOP853 if strict
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9
    )
    
    chief_controlled_states = sol_controlled.y[:6, :].T
    deputy_controlled_states = sol_controlled.y[6:, :].T
    
    # Calculate relative distances
    rel_dist_unc = np.linalg.norm(deputy_unc_states[:, :3] - chief_states[:, :3], axis=1) * 384400.0
    rel_dist_con = np.linalg.norm(deputy_controlled_states[:, :3] - chief_controlled_states[:, :3], axis=1) * 384400.0
    
    # Plotting
    fig = plt.figure(figsize=(14, 6))
    
    # Trajectories
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(chief_states[:, 0], chief_states[:, 1], chief_states[:, 2], 'k-', label='Chief (Ref)')
    ax1.plot(deputy_unc_states[:, 0], deputy_unc_states[:, 1], deputy_unc_states[:, 2], 'r--', label='Uncontrolled')
    ax1.plot(deputy_controlled_states[:, 0], deputy_controlled_states[:, 1], deputy_controlled_states[:, 2], 'g-', label='Controlled (LQR)')
    ax1.set_title('Formation Flying Trajectories')
    ax1.legend()
    
    # Relative Distance
    ax2 = fig.add_subplot(122)
    ax2.plot(sol_chief.t, rel_dist_unc, 'r--', label='Uncontrolled')
    ax2.plot(sol_controlled.t, rel_dist_con, 'g-', label='Controlled')
    ax2.set_xlabel('Time (dimensionless)')
    ax2.set_ylabel('Relative Distance (km)')
    ax2.set_title('Chief-Deputy Separation')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/formation_control.png")
    plt.close()
    print("Saved formation control plot to results/formation_control.png")

def main():
    formation_flight()

if __name__ == "__main__":
    main()
