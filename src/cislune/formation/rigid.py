import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from cislune.core.params import mu, L1_HALO_INITIAL_STATE
from cislune.dynamics.cr3bp import cr3bp_dynamics as halo_dynamics

def constant_formation_flight():
    print("Simulating Multi-Agent (5 Agents) Rigid Formation...")
    
    # Chief initial state (L1 Halo Orbit)
    chief_initial = np.array(L1_HALO_INITIAL_STATE)
    
    # Formation Geometry (Relative to Chief)
    # 10 km in dimensionless units
    dist_km = 10.0
    d = dist_km / 384400.0
    
    # Offsets [x, y, z, vx, vy, vz]
    # Relative velocity must be zero for rigid formation
    offsets = {
        "Deputy 1": np.array([d, 0, 0, 0, 0, 0]),   # +X
        "Deputy 2": np.array([-d, 0, 0, 0, 0, 0]),  # -X
        "Officer 1": np.array([0, 0, d, 0, 0, 0]),  # +Z
        "Officer 2": np.array([0, 0, -d, 0, 0, 0])  # -Z
    }
    
    agent_names = ["Deputy 1", "Deputy 2", "Officer 1", "Officer 2"]
    
    # Initial State Vector Construction
    # [Chief(6), Dep1(6), Dep2(6), Off1(6), Off2(6), DV1(1), DV2(1), DV3(1), DV4(1)]
    # Total size: 6 + 4*6 + 4*1 = 34
    
    y0 = []
    y0.extend(chief_initial)
    for name in agent_names:
        y0.extend(chief_initial + offsets[name])
    y0.extend([0.0] * 4) # Initial Delta-V for each follower
    y0 = np.array(y0)
    
    t_span = (0, 3.0)
    t_eval = np.linspace(0, 3.0, 1000)
    
    def multi_agent_dynamics(t, y):
        # Extract states
        xc = y[:6]
        
        # Chief dynamics
        dxc = halo_dynamics(t, xc)
        
        dy = [0.0] * len(y)
        dy[:6] = dxc
        
        # Process each follower
        for i, name in enumerate(agent_names):
            idx_state = 6 + i*6
            idx_dv = 30 + i
            
            xf = y[idx_state : idx_state+6]
            
            # Natural acceleration at follower position
            dxf_nat = halo_dynamics(t, xf)
            af_nat = np.array(dxf_nat[3:])
            
            # Target acceleration (same as Chief for rigid translation)
            at_target = np.array(dxc[3:])
            
            # Control
            u = at_target - af_nat
            u_mag = np.sqrt(np.sum(u**2))
            
            # State derivative (Kinematics forced to match Chief)
            dy[idx_state : idx_state+3] = dxc[:3] # Velocity matches Chief
            dy[idx_state+3 : idx_state+6] = dxc[3:] # Acceleration matches Chief
            
            # Delta-V accumulation
            dy[idx_dv] = u_mag
            
        return dy

    sol = solve_ivp(
        multi_agent_dynamics,
        t_span,
        y0,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12
    )
    
    # Analysis
    V_star = 1024.0 # m/s
    
    print(f"{'Agent':<15} | {'Total Delta-V (m/s)':<20}")
    print("-" * 40)
    
    final_dvs = sol.y[30:, -1] * V_star
    for name, dv in zip(agent_names, final_dvs):
        print(f"{name:<15} | {dv:.2f}")
        
    # Plotting
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Chief
    cx, cy, cz = sol.y[0], sol.y[1], sol.y[2]
    ax.plot(cx, cy, cz, 'k-', linewidth=2, label='Chief')
    ax.scatter(cx[0], cy[0], cz[0], color='k', marker='o')
    
    # Plot Followers
    colors = ['r', 'g', 'b', 'm']
    for i, name in enumerate(agent_names):
        idx = 6 + i*6
        fx, fy, fz = sol.y[idx], sol.y[idx+1], sol.y[idx+2]
        ax.plot(fx, fy, fz, color=colors[i], linestyle='--', label=f'{name} (DV: {final_dvs[i]:.1f} m/s)')
        # Plot initial point to show formation shape
        ax.scatter(fx[0], fy[0], fz[0], color=colors[i], marker='x')

    ax.set_title('Multi-Agent Rigid Formation (5 Spacecraft)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Zoom in to see formation? No, trajectory is large.
    # Maybe a relative plot would be better?
    # Let's do a relative plot in a separate figure or subplot?
    # For now, just the trajectory.
    
    plt.tight_layout()
    plt.savefig("results/const_formation_multi.png")
    plt.close()
    print("Saved plot to results/const_formation_multi.png")

def main():
    constant_formation_flight()

if __name__ == "__main__":
    main()
