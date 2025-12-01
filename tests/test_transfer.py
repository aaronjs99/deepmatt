import pytest
import numpy as np
from cislune.transfer import optimize_leo_transfer
from unittest.mock import patch, MagicMock

def test_optimize_leo_transfer_runs():
    """
    Test that optimize_leo_transfer runs without error.
    We mock the expensive optimization and integration parts to just verify the flow.
    """
    with patch('cislune.transfer.differential_evolution') as mock_minimize, \
         patch('cislune.transfer.solve_ivp') as mock_solve_ivp, \
         patch('cislune.transfer.get_stable_eigenvector') as mock_get_eig, \
         patch('cislune.transfer.plot_trajectory_3d') as mock_plot:
        
        # Mock minimize result
        mock_res = MagicMock()
        mock_res.success = True
        mock_res.x = [0.0, 1e-5]
        mock_res.fun = 0.1
        mock_minimize.return_value = mock_res
        
        # Mock solve_ivp result
        mock_sol = MagicMock()
        mock_sol.y = np.zeros((6, 10))
        mock_sol.t = np.linspace(0, 1, 10)
        mock_solve_ivp.return_value = mock_sol
        
        # Mock eigenvector
        mock_get_eig.return_value = np.array([1, 0, 0, 0, 0, 0])
        
        # Run function
        optimize_leo_transfer()
        
        # Verify calls
        assert mock_minimize.called
        assert mock_solve_ivp.called
        assert mock_plot.called
