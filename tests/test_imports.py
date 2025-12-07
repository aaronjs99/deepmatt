import pytest


def test_imports():
    """Test that all core deepmatt modules can be imported."""
    from deepmatt.core import params
    from deepmatt.core import utils
    from deepmatt.core import plotting
    from deepmatt.dynamics import cr3bp
    from deepmatt.gnc import navigation
    from deepmatt.gnc.control import SDREController
    from deepmatt.learning import ppo_agent
    from deepmatt.learning import transformer_guidance

    assert True


def test_cli():
    """Test that CLI can be imported."""
    from deepmatt.cli import main

    assert callable(main)
