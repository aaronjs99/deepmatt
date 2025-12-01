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
├── data/                     # Sample data files (e.g., halo_orbit_points.csv)
├── results/                  # Output directory for plots, logs, animations
├── src/cislune/              # Core modules
│   ├── adversary.py          # Optimal control “chase” problem
│   ├── controlled-halo.py    # Controlled halo orbit integration & optimization
│   ├── gravdyn.py            # Gravitational dynamics & GEKKO simulator
│   ├── lissajous.py          # Lissajous orbit point generation
│   ├── littdyn.py            # Linearized–plus–full dynamics simulator
│   ├── oscillators.py        # Nonlinear oscillator examples (van der Pol, Duffing)
│   ├── params.py             # Physical constants & Lagrange points
│   ├── stability.py          # Stability analysis (halo manifold sampling)
│   ├── stationkeeping.py     # Station-keeping MPC for halo orbits
│   ├── uncontrolled-halo.py  # Uncontrolled halo orbit simulation
│   ├── utils.py              # Vector class & integrator utility
├── tests/                    # Unit tests (to add)
├── run.py                    # Entry point for demos
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

# (Optional) Install development dependencies
pip install -e .[dev]
```

## Usage

Run a quick demo of the halo orbit integrator:

```bash
python run.py --mode halo               # uses controlled-halo.py
```

For other demos:

- **Lissajous points:**  
  ```bash
  python run.py --mode lissajous        # uses src/lissajous.py
  ```
- **Stability sampling:**  
  ```bash
  python run.py --mode stability        # uses src/stability.py
  ```
- **Station-keeping MPC:**  
  ```bash
  python run.py --mode stationkeeping   # uses src/stationkeeping.py
  ```
- **Manifold Transfer (Research):**
  ```bash
  python run.py --mode transfer         # uses src/transfer.py
  ```

Output plots, CSVs, and animations will appear under `results/`.

## Known Issues
- The `stationkeeping` mode may fail to converge with the default solver settings.
- `controlled_halo` mode uses the IPOPT solver for better stability.

## Testing

Add and run unit tests via **pytest**:

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