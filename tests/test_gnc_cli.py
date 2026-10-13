"""Exercise the same command entry used by the source and installed launchers."""

import json

import pytest


def test_cli_runs_gnc_with_overrides_and_headless_outputs(tmp_path, capsys):
    from deepmatt.cli import main

    settings = tmp_path / "run.yaml"
    settings.write_text("duration: 1\ntime_step: 0.1\nseed: 99", encoding="utf-8")
    main(
        [
            "--mode",
            "gnc",
            "--config",
            str(settings),
            "--duration",
            "0.025",
            "--dt",
            "0.01",
            "--seed",
            "7",
            "--output-dir",
            str(tmp_path / "runs"),
            "--no-plot",
        ]
    )
    (run_dir,) = (tmp_path / "runs").iterdir()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["duration"] == 0.025
    assert config["time_step"] == 0.01
    assert config["seed"] == 7
    assert "completed" in capsys.readouterr().out.lower()


@pytest.mark.parametrize(
    "args",
    [
        ["--mode", "gnc", "--duration", "0"],
        ["--mode", "gnc", "--dt", "nan"],
        ["--mode", "gnc", "--seed", "-1"],
        ["--mode", "stability", "--duration", "1"],
        ["--mode", "formation", "--no-plot"],
    ],
)
def test_cli_rejects_invalid_or_unsupported_options_without_traceback(args, capsys):
    from deepmatt.cli import main

    with pytest.raises(SystemExit) as failure:
        main(args)
    assert failure.value.code == 2
    error = capsys.readouterr().err
    assert "Traceback" not in error


def test_cli_reports_a_missing_config_without_creating_output(tmp_path, capsys):
    from deepmatt.cli import main

    output = tmp_path / "runs"
    with pytest.raises(SystemExit) as failure:
        main(
            [
                "--mode",
                "gnc",
                "--config",
                str(tmp_path / "missing.yaml"),
                "--output-dir",
                str(output),
            ]
        )
    assert failure.value.code == 2
    assert not output.exists()
    assert "missing.yaml" in capsys.readouterr().err


@pytest.mark.parametrize(
    "failure,code",
    [(RuntimeError("test solver failed"), 1), (KeyboardInterrupt(), 130)],
)
def test_cli_failure_and_cancellation_do_not_report_success(
    tmp_path, capsys, monkeypatch, failure, code
):
    from deepmatt.cli import main
    import deepmatt.gnc.integrated_sim as simulation

    def fail_run(config):
        raise failure

    monkeypatch.setattr(simulation, "run_gnc_simulation", fail_run)
    output = tmp_path / "runs"
    with pytest.raises(SystemExit) as stopped:
        main(["--mode", "gnc", "--output-dir", str(output)])
    assert stopped.value.code == code
    message = capsys.readouterr()
    assert "completed" not in message.out.lower()
    assert "Traceback" not in message.err
    assert not output.exists()
