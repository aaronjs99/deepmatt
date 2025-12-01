import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cislune.params import mu, EARTH_POS, MOON_POS

def plot_trajectory_3d(states, title, filename, show_earth=True, show_moon=True, other_points=None):
    """
    Plots a 3D trajectory.
    
    Args:
        states: Nx3 or Nx6 array of state vectors.
        title: Plot title.
        filename: Output filename (saved to results/).
        show_earth: Boolean to show Earth.
        show_moon: Boolean to show Moon.
        other_points: List of dicts with keys 'pos', 'color', 'marker', 's', 'label'.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(states[:, 0], states[:, 1], states[:, 2], 'b-', label='Trajectory')
    
    if show_earth:
        ax.scatter([EARTH_POS[0]], [EARTH_POS[1]], [EARTH_POS[2]], color='blue', s=100, label='Earth')
        
    if show_moon:
        ax.scatter([MOON_POS[0]], [MOON_POS[1]], [MOON_POS[2]], color='gray', s=50, label='Moon')
        
    if other_points:
        for point in other_points:
            ax.scatter([point['pos'][0]], [point['pos'][1]], [point['pos'][2]], 
                       color=point.get('color', 'red'), 
                       marker=point.get('marker', 'o'), 
                       s=point.get('s', 20), 
                       label=point.get('label', 'Point'))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    # Ensure results directory exists
    import os
    if not os.path.exists("results"):
        os.makedirs("results")
        
    filepath = f"results/{filename}"
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot to {filepath}")
