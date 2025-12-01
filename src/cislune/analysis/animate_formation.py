import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from cislune.formation.rigid import constant_formation_flight

def animate_formation():
    print("Generating Formation Animation...")
    
    # Run simulation to get data
    sol = constant_formation_flight()
    
    # Extract data
    t = sol.t
    # Chief: 6-8
    cx, cy, cz = sol.y[6], sol.y[7], sol.y[8]
    
    agent_names = ["Deputy 1", "Deputy 2", "Officer 1", "Officer 2"]
    followers = []
    for i in range(4):
        idx = 12 + i*6
        fx, fy, fz = sol.y[idx], sol.y[idx+1], sol.y[idx+2]
        followers.append((fx, fy, fz))
        
    # Setup Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Limits
    all_x = np.concatenate([cx] + [f[0] for f in followers])
    all_y = np.concatenate([cy] + [f[1] for f in followers])
    all_z = np.concatenate([cz] + [f[2] for f in followers])
    
    ax.set_xlim(np.min(all_x), np.max(all_x))
    ax.set_ylim(np.min(all_y), np.max(all_y))
    ax.set_zlim(np.min(all_z), np.max(all_z))
    
    ax.set_xlabel('X (Rotating)')
    ax.set_ylabel('Y (Rotating)')
    ax.set_zlabel('Z (Rotating)')
    ax.set_title('Robust Formation Flight (Station Keeping + Collision Avoidance)')
    
    # Lines and Dots
    # Chief
    line_c, = ax.plot([], [], [], 'k-', linewidth=1, alpha=0.5)
    dot_c, = ax.plot([], [], [], 'ko', label='Chief')
    
    # Followers
    lines_f = []
    dots_f = []
    colors = ['r', 'g', 'b', 'm']
    
    for i, name in enumerate(agent_names):
        l, = ax.plot([], [], [], color=colors[i], linestyle='--', linewidth=0.5, alpha=0.5)
        d, = ax.plot([], [], [], color=colors[i], marker='o', label=name)
        lines_f.append(l)
        dots_f.append(d)
        
    ax.legend()
    
    # Animation Function
    # Downsample for speed
    skip = 5
    num_frames = len(t) // skip
    
    def update(frame):
        idx = frame * skip
        
        # Update Chief
        line_c.set_data(cx[:idx], cy[:idx])
        line_c.set_3d_properties(cz[:idx])
        dot_c.set_data([cx[idx]], [cy[idx]])
        dot_c.set_3d_properties([cz[idx]])
        
        # Update Followers
        for i in range(4):
            fx, fy, fz = followers[i]
            lines_f[i].set_data(fx[:idx], fy[:idx])
            lines_f[i].set_3d_properties(fz[:idx])
            dots_f[i].set_data([fx[idx]], [fy[idx]])
            dots_f[i].set_3d_properties([fz[idx]])
            
        return line_c, dot_c, *lines_f, *dots_f
        
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=20, blit=False)
    
    save_path = "results/formation/formation_animation.gif"
    print(f"Saving animation to {save_path} (this might take a while)...")
    ani.save(save_path, writer='pillow', fps=30)
    print("Animation saved!")

def main():
    animate_formation()

if __name__ == "__main__":
    main()
