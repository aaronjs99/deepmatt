"""Regression coverage for a reproducible, inspectable GNC experiment."""

import json

import numpy as np
import pytest
from scipy.integrate import solve_ivp


def make_config(**overrides):
    from deepmatt.gnc.config import GNCConfig

    return GNCConfig(duration=0.025, time_step=0.01, seed=7, **overrides)


@pytest.mark.parametrize(
    "values",
    [
        {"duration": 0},
        {"duration": float("nan")},
        {"duration": 10**400},
        {"time_step": -0.1},
        {"time_step": float("inf")},
        {"seed": -1},
        {"seed": True},
        {"measurement_std": 0},
        {"measurement_std": 1e-300},
        {"initial_estimate_std": 1e300},
        {"thruster_noise_std": -1},
        {"max_acceleration": 0},
        {"chief_initial_state": [1, 2, 3]},
        {"relative_offset": [0, 0, 0, 0, 0, float("nan")]},
        {"lqr_state_weights": [1, 1, 1, 1, 1, -1]},
    ],
)
def test_config_rejects_invalid_inputs(values):
    from deepmatt.gnc.config import GNCConfig

    with pytest.raises(ValueError):
        GNCConfig(**values)


def test_config_rejects_unknown_keys_and_bad_yaml(tmp_path):
    from deepmatt.gnc.config import load_gnc_config

    path = tmp_path / "settings.yaml"
    for content in (
        "duraton: 1",
        "- duration",
        "duration: [",
        "duration: 1\nduration: 2",
    ):
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            load_gnc_config(path)


def test_config_replays_json_and_normalizes_numeric_yaml(tmp_path):
    from deepmatt.gnc.config import load_gnc_config

    path = tmp_path / "settings.yaml"
    path.write_text("duration: 0.025\nmeasurement_std: 1e-8\nseed: 7", encoding="utf-8")
    config = load_gnc_config(path)
    replay = tmp_path / "config.json"
    replay.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    assert load_gnc_config(replay) == config
    assert config.measurement_std == 1e-8


def test_short_run_reaches_final_time_and_records_final_physics():
    from deepmatt.dynamics.cr3bp import cr3bp_dynamics
    from deepmatt.gnc.integrated_sim import run_gnc_simulation

    result = run_gnc_simulation(make_config())
    np.testing.assert_allclose(result.time, [0, 0.01, 0.02, 0.025], atol=1e-15)
    assert result.chief_state.shape == (4, 6)
    assert result.control_applied.shape == (3, 3)
    expected = solve_ivp(
        cr3bp_dynamics,
        (0, 0.025),
        result.chief_state[0],
        rtol=1e-11,
        atol=1e-13,
    ).y[:, -1]
    np.testing.assert_allclose(result.chief_state[-1], expected, atol=1e-10)
    expected_delta_v = np.sum(
        np.linalg.norm(result.control_applied, axis=1) * [0.01, 0.01, 0.005]
    )
    assert result.delta_v[-1] == pytest.approx(expected_delta_v)


def test_seeded_run_is_repeatable_without_changing_global_random_state():
    from deepmatt.gnc.integrated_sim import run_gnc_simulation

    before = np.random.get_state()
    first = run_gnc_simulation(make_config())
    second = run_gnc_simulation(make_config())
    after = np.random.get_state()
    np.testing.assert_array_equal(first.deputy_state, second.deputy_state)
    np.testing.assert_array_equal(first.control_applied, second.control_applied)
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    other = run_gnc_simulation(make_config(initial_estimate_std=1e-5))
    assert not np.array_equal(first.deputy_state, other.deputy_state)


def test_position_metrics_exclude_velocity_and_actuator_limit_is_respected():
    from deepmatt.gnc.integrated_sim import run_gnc_simulation

    result = run_gnc_simulation(
        make_config(
            initial_relative_error=(0, 0, 0, 0.001, 0, 0), max_acceleration=1e-5
        )
    )
    assert result.position_error[0] == pytest.approx(0, abs=1e-15)
    assert result.velocity_error[0] == pytest.approx(0.001)
    assert np.max(np.linalg.norm(result.control_applied, axis=1)) <= 1.000001e-5
    assert np.all(np.isfinite(result.estimated_relative_state))
    assert np.linalg.eigvalsh(result.covariance).min() >= -1e-20


def test_custom_system_changes_dynamics_without_inventing_physical_units():
    from deepmatt.core.system import CR3BPSystem
    from deepmatt.gnc.integrated_sim import run_gnc_simulation

    custom = run_gnc_simulation(make_config(system=CR3BPSystem("synthetic", 0.2)))
    earth_moon = run_gnc_simulation(make_config())
    assert not np.allclose(custom.chief_state[-1], earth_moon.chief_state[-1])
    assert custom.summary()["final_position_error_m"] is None
    assert custom.summary()["total_delta_v_m_s"] is None


def test_deadband_applies_to_the_limited_command_known_by_navigation():
    from deepmatt.gnc.integrated_sim import run_gnc_simulation

    result = run_gnc_simulation(
        make_config(
            initial_relative_error=(0.1, 0, 0, 0, 0, 0),
            max_acceleration=0.001,
            thruster_deadband=0.01,
        )
    )
    assert np.linalg.norm(result.control_requested[0]) > 0.01
    np.testing.assert_array_equal(result.control_command, np.zeros((3, 3)))
    np.testing.assert_array_equal(result.control_applied, np.zeros((3, 3)))


def test_output_bundle_is_replayable_and_repeated_saves_do_not_overwrite(tmp_path):
    from deepmatt.gnc.config import load_gnc_config
    from deepmatt.gnc.integrated_sim import run_gnc_simulation
    from deepmatt.output.gnc import save_gnc_result

    result = run_gnc_simulation(make_config())
    first = save_gnc_result(result, tmp_path, plot=False)
    second = save_gnc_result(result, tmp_path, plot=False)
    assert first != second
    assert load_gnc_config(first / "config.json") == result.config
    summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["sample_count"] == 4
    assert summary["total_delta_v_m_s"] >= 0
    table = np.genfromtxt(first / "trajectory.csv", delimiter=",", names=True)
    np.testing.assert_allclose(table["time_nd"], result.time)
    with np.load(first / "states.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["deputy_state"], result.deputy_state)
        np.testing.assert_array_equal(saved["control_applied"], result.control_applied)
    assert not (first / "performance.png").exists()


@pytest.mark.parametrize(
    "failure", [OSError("test disk full"), KeyboardInterrupt("test interrupted")]
)
def test_output_failure_does_not_leave_a_completed_summary(
    tmp_path, monkeypatch, failure
):
    from deepmatt.gnc.integrated_sim import run_gnc_simulation
    from deepmatt.output.gnc import save_gnc_result

    def disk_full(*args, **kwargs):
        raise failure

    monkeypatch.setattr(np, "savez_compressed", disk_full)
    with pytest.raises(type(failure), match="test"):
        save_gnc_result(run_gnc_simulation(make_config()), tmp_path, plot=False)
    summaries = list(tmp_path.glob("*/summary.json"))
    assert len(summaries) == 1
    assert (
        json.loads(summaries[0].read_text(encoding="utf-8"))["status"] == "incomplete"
    )


def test_solver_failure_is_reported_instead_of_using_partial_output(monkeypatch):
    import deepmatt.gnc.integrated_sim as simulation

    class FailedSolution:
        success = False
        message = "test integration failed"

    monkeypatch.setattr(
        simulation, "solve_ivp", lambda *args, **kwargs: FailedSolution()
    )
    with pytest.raises(RuntimeError, match="integration failed"):
        simulation.run_gnc_simulation(make_config())
