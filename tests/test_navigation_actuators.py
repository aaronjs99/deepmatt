"""Numerical regressions required by the configurable GNC runner."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp


def test_navigation_prediction_matches_frozen_linear_model():
    from deepmatt.core.params import L1_HALO_INITIAL_STATE
    from deepmatt.dynamics.cr3bp import linearized_dynamics
    from deepmatt.gnc.navigation import EKF

    initial = np.array([1e-4, -2e-4, 1e-4, 1e-5, 2e-5, -1e-5])
    control = np.array([1e-3, -2e-3, 1e-3])
    dynamics = linearized_dynamics(L1_HALO_INITIAL_STATE)
    input_term = np.r_[np.zeros(3), control]
    expected = solve_ivp(
        lambda t, x: dynamics @ x + input_term,
        (0, 0.1),
        initial,
        rtol=1e-12,
        atol=1e-14,
    ).y[:, -1]
    ekf = EKF(initial, np.eye(6) * 1e-8, np.zeros((6, 6)), np.eye(3) * 1e-10)
    ekf.predict(control, 0.1, L1_HALO_INITIAL_STATE)
    np.testing.assert_allclose(ekf.x, expected, rtol=1e-9, atol=1e-12)


def test_navigation_keeps_measurement_uncertainty_at_large_covariance():
    from deepmatt.gnc.navigation import EKF

    ekf = EKF(np.zeros(6), np.eye(6) * 1e20, np.zeros((6, 6)), np.eye(3))
    ekf.update(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(ekf.x[:3], [1, 2, 3])
    np.testing.assert_allclose(np.diag(ekf.P)[:3], [1, 1, 1])
    np.testing.assert_allclose(ekf.P, ekf.P.T)


def test_navigation_owns_initial_arrays_and_rejects_invalid_step():
    from deepmatt.core.params import L1_HALO_INITIAL_STATE
    from deepmatt.gnc.navigation import EKF

    state = np.zeros(6)
    covariance = np.eye(6)
    ekf = EKF(state, covariance, np.eye(6) * 0.1, np.eye(3))
    state[0] = 99
    covariance[0, 0] = 99
    assert ekf.x[0] == 0
    assert ekf.P[0, 0] == 1
    for step in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            ekf.predict(np.zeros(3), step, L1_HALO_INITIAL_STATE)


def test_thruster_seed_is_local_and_caps_actual_acceleration_after_noise():
    from deepmatt.gnc.actuators import Thruster

    first = Thruster(
        noise_std=0.1, bias=0.2, max_acceleration=0.05, rng=np.random.default_rng(12)
    )
    second = Thruster(
        noise_std=0.1, bias=0.2, max_acceleration=0.05, rng=np.random.default_rng(12)
    )
    before = np.random.get_state()
    actual = first.fire(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(actual, second.fire(np.array([1.0, 2.0, 3.0])))
    assert np.linalg.norm(actual) == pytest.approx(0.05)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    np.testing.assert_array_equal(first.fire(np.zeros(3)), np.zeros(3))


@pytest.mark.parametrize(
    "values",
    [
        {"noise_std": -1},
        {"bias": float("nan")},
        {"min_impulse": -1},
        {"max_acceleration": 0},
    ],
)
def test_thruster_rejects_invalid_parameters(values):
    from deepmatt.gnc.actuators import Thruster

    with pytest.raises(ValueError):
        Thruster(**values)
