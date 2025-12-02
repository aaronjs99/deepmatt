# CisLune: Stable Control in Cislunar Space

CisLune explores cislunar dynamics and control, offering a suite of scripts to:
- Model the restricted three-body problem (Earth–Moon–spacecraft)  
- Simulate and visualize halo and Lissajous orbits  
- Design and optimize station-keeping maneuvers  
- Analyze dynamical stability and manifold structures  

## Poincaré Maps

A Poincaré map is a powerful tool for analyzing the dynamics of high-dimensional systems, such as the Circular Restricted Three-Body Problem (CR3BP). Instead of viewing the entire continuous trajectory in the 6D state space (position and velocity), we look at the intersection of the trajectory with a lower-dimensional subspace, called the Poincaré section.

In this project, we use the $z=0$ plane as our section. Every time a trajectory crosses this plane (with $v_z$ having a specific sign, or both), we plot its state $(x, \dot{x})$.

**Why is this useful?**
1.  **Dimensionality Reduction**: It reduces the problem from continuous time to a discrete map, making it easier to visualize structure.
2.  **Stability Analysis**: Fixed points on the map correspond to periodic orbits. The behavior of points near a fixed point reveals the stability of that orbit.
3.  **Manifold Structure**: By plotting many trajectories on the stable manifold of a Halo orbit, we can see the "tube" structure that leads into the orbit. This is crucial for designing low-energy transfers.


## Repository Structure

```
.
├── config/
│   └── default.yaml          # Simulation configuration files
├── data/                     # Sample data files
├── models/                   # Trained RL models (PPO, Transformer)
├── paper/                    # LaTeX source for the conference paper
├── results/                  # Output directory for plots, logs, animations
├── src/cislune/              # Core modules
│   ├── analysis/             # Analysis scripts (halo, lissajous, stability, transfer)
│   ├── cli.py                # Main CLI entry point
│   ├── core/                 # Core utilities (animator, plotting, params)
│   ├── dynamics/             # CR3BP dynamics implementation
│   ├── formation/            # Formation flying logic
│   ├── gnc/                  # Guidance, Navigation, and Control (EKF, LQR, SDRE)
│   ├── learning/             # Deep RL and Transformer agents
│   └── validation/           # Benchmarking and validation scripts
├── tests/                    # Unit tests
├── run.py                    # Wrapper for the CLI
├── pyproject.toml
├── .gitignore
├── README.md
└── LICENSE
```

## Installation

Requires **Python 3.11** (or any Python ≥3.9).

```bash
# Clone the repository
git clone https://github.com/aaronjs99/cislune
cd cislune

# Create a Conda environment
conda create -n cislune python=3.11 -y
conda activate cislune

# Install in editable mode
pip install -e .

# Install dependencies
pip install numpy scipy matplotlib gekko stable-baselines3 gymnasium torch tensorboard
```

## Usage

The project uses a unified CLI. You can run different modes using `run.py`.

### 1. Generate All Results (Paper Reproduction)
To generate all plots and animations used in the paper:
```bash
python run.py --mode generate_results
```
This will populate the `results/` directory with:
- `results/analysis/`: Halo orbits, manifolds, transfers
- `results/formation/`: Formation flying animations and plots
- `results/validation/`: Monte Carlo and benchmark plots

### 2. Run Specific Modules
- **Halo Orbit Analysis:**
  ```bash
  python run.py --mode halo
  ```
- **Lissajous Orbits:**
  ```bash
  python run.py --mode lissajous
  ```
- **Stability Analysis:**
  ```bash
  python run.py --mode stability
  ```
- **Formation Flying Simulation:**
  ```bash
  python run.py --mode formation
  ```

## Testing

Run unit tests via **pytest**:

```bash
pytest tests/ -v
```

## Contributing

Contributions are welcome! Please:

1. Fork the repo  
2. Create a feature branch (`git checkout -b feature/YourFeature`)  
3. Commit your changes (`git commit -m "Add X"`)  
4. Push (`git push origin feature/YourFeature`)  
5. Open a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for more.

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.