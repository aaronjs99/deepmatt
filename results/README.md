# DeepMAtt Simulation Results

This directory contains the output plots and data from the `deepmatt` simulation package.

## Orbit Simulations
- **`uncontrolled_halo_3d.png`**: Visualization of a natural L1 Halo orbit. Note the instability causing the spacecraft to depart the vicinity of L1 over time.
- **`controlled_halo_3d.png`**: A station-keeping simulation where control inputs maintain the spacecraft in the Halo orbit.
- **`lissajous_3d.png`**: A quasi-periodic Lissajous orbit around L1.

## Transfer Trajectories
- **`transfer_earth.png`**: An optimized transfer trajectory using the stable manifold to approach Earth from the L1 Halo orbit.
- **`transfer_leo.png`**: A refined transfer trajectory that targets a specific Low Earth Orbit (LEO) radius (~6700 km) for insertion.
- **`transfer_fast.png`**: A time-optimal transfer trajectory that minimizes the flight duration while maintaining a close approach to Earth.

## Research Analysis
- **`poincare_map.png`**: A Poincaré section ($z=0$ plane) of the stable manifold, revealing the phase space structure and stability regions.
- **`formation_flying.png`**: Simulation of a two-spacecraft formation (Chief and Deputy) in a Halo orbit, showing relative drift without control.
- **`adversary_pareto.png`**: (If applicable) Pareto front analysis for adversarial scenarios in cislunar space.

## Data
- **`halo_orbit_points.csv`**: Raw state data for the generated Halo orbit.
