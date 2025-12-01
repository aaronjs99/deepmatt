import pytest
import numpy as np

from cislune.params import mu_earth_moon

from cislune.utils import integrate_simulator, Vector
from cislune.params import mu_earth_moon

def test_integrator():
    # Simple harmonic oscillator in X: x'' = -x
    def sys(t, pos, vel):
        return Vector(-pos.x, 0.0, 0.0)
    
    p0 = Vector(1.0, 0.0, 0.0)
    v0 = Vector(0.0, 0.0, 0.0)
    
    # Integrate for 2*pi period
    dt = 0.01
    num_steps = int(2 * np.pi / dt)
    
    pos_list, vel_list = integrate_simulator(sys, p0, v0, dt=dt, num_steps=num_steps)
    
    final_pos = pos_list[-1]
    final_vel = vel_list[-1]
    
    # Should be back at (1, 0, 0)
    assert np.isclose(final_pos.x, 1.0, atol=1e-1)
    assert np.isclose(final_pos.y, 0.0, atol=1e-1)
    assert np.isclose(final_vel.x, 0.0, atol=1e-1)

def test_params():
    assert mu_earth_moon > 0
    assert mu_earth_moon < 1
