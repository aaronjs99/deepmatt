# /***********************************************************
# *                                                         *
# * Copyright (c) 2025                                      *
# *                                                         *
# * University of California, Los Angeles                   *
# *                                                         *
# * Author : Aaron John Sabu                                *
# * Contact: aaronjs@g.ucla.edu                             *
# *                                                         *
# ***********************************************************/

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gekko import GEKKO


class ChaseProblem:
    """
    Encapsulates a single chase optimization using GEKKO.

    Attributes:
        init_pos: List of (position, velocity, acceleration) tuples for target and chaser.
        bounds: Control bounds for target ('T') and chaser ('C').
        k1: Weight on final time in the objective function.
        time_steps: Number of discretization points in the time horizon.
        remote: Whether to use the remote GEKKO server.
    """

    def __init__(
        self,
        init_pos: List[Tuple[float, float, float]],
        bounds: Dict[str, Tuple[float, float]],
        k1: float,
        time_steps: int,
        remote: bool,
    ):
        """
        Initialize a ChaseProblem instance.

        Args:
            init_pos: Initial (pos, vel, acc) for target and chaser.
            bounds: Control bounds for 'T' (target) and 'C' (chaser).
            k1: Weight on final time in objective.
            time_steps: Number of time discretization points.
            remote: Use remote GEKKO server if True.
        """
        self.init_pos = init_pos
        self.bounds = bounds
        self.k1 = k1
        self.time_steps = time_steps
        self.remote = remote

    def solve(self, k2: float) -> Tuple[float, float]:
        """
        Solve the optimal control problem for a given k2 weight.

        Args:
            k2: Weight on chaser control effort in objective.

        Returns:
            final_time: Optimal final time.
            control_effort: Sum of squared chaser controls.
        """
        m = GEKKO(remote=self.remote, server="http://apmonitor.com")
        m.time = np.linspace(0, 1, self.time_steps)

        # Free final time variable
        T = m.FV(value=1.0, lb=1e-10, ub=1e10)
        T.STATUS = 1

        # States for target (T) and chaser (C)
        states: Dict[str, Tuple[Any, Any, Any]] = {}
        for agent, init in zip(["T", "C"], self.init_pos):
            x = m.Var(value=init[0], fixed_initial=True, name=f"x_{agent}")
            v = m.Var(value=init[1], fixed_initial=True, name=f"v_{agent}")
            a = m.Var(value=init[2], fixed_initial=True, name=f"a_{agent}")
            states[agent] = (x, v, a)

        # Control inputs
        u_T = m.Var(name="u_T")
        u_C = m.Var(lb=self.bounds["C"][0], ub=self.bounds["C"][1], name="u_C")

        # Dynamics equations
        m.Equation(states["T"][0].dt() / T == states["T"][1])
        m.Equation(states["T"][1].dt() / T == states["T"][2])
        m.Equation(states["T"][2].dt() / T == u_T)
        m.Equation(states["C"][0].dt() / T == states["C"][1])
        m.Equation(states["C"][1].dt() / T == states["C"][2])
        m.Equation(states["C"][2].dt() / T == u_C)

        # Target control law
        m.Equation(
            u_T == self.bounds["T"][1] * m.tanh(100 * (states["T"][0] - states["C"][0]))
        )

        # Mask for integrator and final time
        integrator_mask = np.zeros_like(m.time)
        final_mask = np.zeros_like(m.time)
        final_mask[-1] = 1.0
        integrator = m.Param(value=integrator_mask)
        final = m.Param(value=final_mask)
        
        # Terminal catch constraint at final time
        m.Equation((states["T"][0] - states["C"][0]) * final <= 1e-3)

        # Objective: weighted time + chaser control effort
        m.Minimize(final * (self.k1 * T) + integrator * (k2 * u_C**2))

        # Solver options
        m.options.IMODE = 6
        m.options.SOLVER = 3
        m.options.MAX_ITER = 1000

        # Solve
        try:
            m.solve(disp=True)
        except Exception as e:
            print(f"Solver failed: {e}")
            raise

        final_time = T.value[0]
        control_effort = sum(u**2 for u in u_C.value)
        return final_time, control_effort


class ParetoSweep:
    """
    Perform a parameter sweep over multiple k2 values to build a Pareto front.

    Attributes:
        init_pos: Initial state positions for target and chaser.
        bounds: Control bounds dictionary.
        k1: Time weight.
        k2_values: Array of k2 weights to sweep.
        time_steps: Discretization points.
        remote: Use remote solver flag.
    """
    def __init__(
        self,
        init_pos: List[Tuple[float, float, float]],
        bounds: Dict[str, Tuple[float, float]],
        k1: float,
        k2_values: np.ndarray,
        time_steps: int,
        remote: bool,
    ):
        """
        Initialize a ParetoSweep instance.

        Args:
            init_pos: Initial (pos, vel, acc) tuples.
            bounds: Control bounds.
            k1: Time weight.
            k2_values: Sequence of control weights to evaluate.
            time_steps: Number of time discretizations.
            remote: Remote solver flag.
        """
        self.init_pos = init_pos
        self.bounds = bounds
        self.k1 = k1
        self.k2_values = k2_values
        self.time_steps = time_steps
        self.remote = remote

    def run(self) -> pd.DataFrame:
        """
        Execute the sweep over all k2 values.

        Returns:
            DataFrame with columns ['k2', 'time', 'effort'] of results.
        """
        records: List[Dict[str, Any]] = []
        problem = ChaseProblem(
            self.init_pos, self.bounds, self.k1, self.time_steps, self.remote
        )
        for k2 in self.k2_values:
            try:
                tf, eff = problem.solve(k2)
                print(f"k2={k2:.3f}: time={tf:.4f}, effort={eff:.4f}")
                records.append({"k2": k2, "time": tf, "effort": eff})
            except Exception as e:
                print(f"Solver failed at k2={k2:.3f}: {e}")
        return pd.DataFrame(records)

    @staticmethod
    def plot(df: pd.DataFrame) -> None:
        """
        Plot the Pareto front (time vs effort) from a DataFrame.

        Args:
            df: DataFrame with 'time' and 'effort' columns.
        """
        plt.figure()
        plt.plot(df["time"], df["effort"], "o-")
        plt.xlabel("Time")
        plt.ylabel("Control Effort")
        plt.title("Pareto Front")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def main(
    k1: float = 1.0,
    k2_start: float = 0.0,
    k2_stop: float = 1.0,
    steps: int = 21,
    time_steps: int = 201,
    remote: bool = True,
    plot: bool = True,
) -> None:
    """
    Run an adversary Pareto sweep and save results.

    Args:
        k1: Weight on final time.
        k2_start: Starting control weight.
        k2_stop: Ending control weight.
        steps: Number of k2 samples.
        time_steps: Discretization points.
        remote: Use remote GEKKO server if True.
        plot: Show plot if True.
    """
    init_positions = [(1.0, 0.5, 0.5), (0.0, 0.0, 0.0)]
    control_bounds = {"T": (-1.0, 1.0), "C": (-2.0, 2.0)}

    k2_vals = np.linspace(k2_start, k2_stop, steps)
    sweep = ParetoSweep(
        init_positions, control_bounds, k1, k2_vals, time_steps, remote=remote
    )
    df = sweep.run()

    output_dir = Path("./results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "adversary_pareto.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved Pareto data to {output_path}")

    if plot and not df.empty:
        ParetoSweep.plot(df)


if __name__ == "__main__":
    main()
