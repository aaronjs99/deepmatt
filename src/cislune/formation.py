import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from cislune.params import mu, L1_HALO_INITIAL_STATE
from cislune.gravdyn import cr3bp_dynamics as halo_dynamics

def formation_flight():
    print("Simulating Formation Flying in Halo Orbit...")
    
    # Chief initial state (L1 Halo Orbit)
    chief_initial = np.array(L1_HALO_INITIAL_STATE)
    
    # Deputy initial state (Perturbed from Chief)
    # Let's say Deputy is 10 km away in X direction initially
    # 10 km in dimensionless units: 10 / 384400
    dist_dim = 10.0 / 384400.0
    deputy_initial = chief_initial.copy()
    deputy_initial[0] += dist_dim
    
    t_span = (0, 3.0) # Approx one period
    t_eval = np.linspace(0, 3.0, 1000)
    
    # Integrate Chief
    sol_chief = solve_ivp(
        halo_dynamics,
        t_span,
        chief_initial,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12
    )
    
    # Integrate Deputy (Natural dynamics for now, no control)
    sol_deputy = solve_ivp(
        halo_dynamics,
        t_span,
        deputy_initial,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12
    )
    
    chief_state = sol_chief.y.T
    deputy_state = sol_deputy.y.T
    
    # Calculate relative distance
    rel_dist = np.linalg.norm(deputy_state[:, :3] - chief_state[:, :3], axis=1) * 384400.0 # km
    
    # Plot Trajectories
    fig = plt.figure(figsize=(12, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(chief_state[:, 0], chief_state[:, 1], chief_state[:, 2], 'b-', label='Chief')
    ax1.plot(deputy_state[:, 0], deputy_state[:, 1], deputy_state[:, 2], 'r--', label='Deputy')
    ax1.set_title('Formation Flying Trajectories')
    ax1.legend()
    
    # Plot Relative Distance
    ax2 = fig.add_subplot(122)
    ax2.plot(sol_chief.t, rel_dist)
    ax2.set_xlabel('Time (dimensionless)')
    ax2.set_ylabel('Relative Distance (km)')
    ax2.set_title('Chief-Deputy Separation (Uncontrolled)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/formation_flying.png")
    plt.close()
    print("Saved formation plot to results/formation_flying.png")

def main():
    formation_flight()

if __name__ == "__main__":
    main()
