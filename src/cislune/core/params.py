# Physical constants
mu = 0.01215058560962404  # Mass parameter for Earth-Moon system (dimensionless)

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
EARTH_POS = [-mu, 0, 0]
MOON_POS = [1 - mu, 0, 0]
