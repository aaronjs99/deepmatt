# /***********************************************************
# *                                                         *
# * Copyright (c) 2025                                      *
# *                                                         *
# * University of California, Los Angeles                   *
# *                                                         *
# * Author : Aaron John-Sabu                                *
# * Contact: aaronjs@g.ucla.edu                             *
# *                                                         *
# ***********************************************************/

"""Legacy Earth--Moon constants.

New code should accept a :class:`deepmatt.core.system.CR3BPSystem` explicitly.
These aliases remain for existing notebooks and analysis scripts.
"""

from deepmatt.core.system import EARTH_MOON_SYSTEM

# Legacy aliases for existing scripts.
mu = EARTH_MOON_SYSTEM.mass_parameter

# Initial State for L1 Halo Orbit (Reference)
L1_HALO_INITIAL_STATE = [
    8.2352859616538130e-01,
    6.3030050026113780e-03,
    1.1041843760609028e-06,
    5.4958758023583450e-03,
    1.2551080993197510e-01,
    -2.4522163866711757e-07,
]

# Celestial Body Positions (Dimensionless)
EARTH_POS = list(EARTH_MOON_SYSTEM.primary_position)
MOON_POS = list(EARTH_MOON_SYSTEM.secondary_position)

# Characteristic Quantities
L_star = EARTH_MOON_SYSTEM.length_scale_km  # Characteristic length (km)
T_star = EARTH_MOON_SYSTEM.time_scale_s  # Characteristic time (s)
