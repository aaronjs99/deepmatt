"""Keep applicable core/transfer coverage from the retired aggregate tests."""

from types import SimpleNamespace

import numpy as np
import pytest


def test_vector_integrator_tracks_oscillation_without_mutating_initial_state():
    from deepmatt.core.utils import Vector, integrate_simulator

    position, velocity = Vector(1, 0, 0), Vector(0, 0, 0)
    positions, velocities = integrate_simulator(
        lambda t, p, v: Vector(-p.x, 0, 0),
        position,
        velocity,
        dt=0.01,
        num_steps=628,
    )
    assert positions[-1].x == pytest.approx(np.cos(6.28), abs=0.01)
    assert velocities[-1].x == pytest.approx(-np.sin(6.28), abs=0.01)
    assert position.x == 1 and velocity.x == 0


@pytest.mark.parametrize("mass", [0, 1, float("nan")])
def test_system_rejects_nonphysical_mass_fraction(mass):
    from deepmatt.core.system import CR3BPSystem

    with pytest.raises(ValueError):
        CR3BPSystem("invalid", mass)


def test_leo_transfer_reports_selected_solution_and_computed_injection_cost(
    monkeypatch,
):
    from deepmatt.analysis import transfer
    from deepmatt.core.params import mu

    radius = 6700 / 384400
    x = radius - mu
    circular_velocity = np.sqrt((1 - mu) / radius)
    # A circular perigee fixture gives zero injection delta-V under this
    # legacy routine's inertial-velocity conversion.
    state = np.array([x, 0, 0, 0, circular_velocity - x, 0])
    trajectory = np.repeat(state[:, None], 10, axis=1)
    monkeypatch.setattr(
        transfer,
        "solve_ivp",
        lambda *a, **k: SimpleNamespace(y=trajectory.copy(), t=np.linspace(0, 1, 10)),
    )
    monkeypatch.setattr(
        transfer,
        "differential_evolution",
        lambda *a, **k: SimpleNamespace(success=True, x=[0.25, 1e-5], fun=0.0),
    )
    monkeypatch.setattr(transfer, "plot_trajectory_3d", lambda *a, **k: None)
    tau, offset, delta_v = transfer.optimize_leo_transfer()
    assert tau == 0.25
    assert offset == 1e-5
    assert delta_v == pytest.approx(0, abs=1e-12)
