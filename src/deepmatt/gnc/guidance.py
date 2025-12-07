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

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eig

from deepmatt.core.params import mu, L1_HALO_INITIAL_STATE
from deepmatt.dynamics.cr3bp import cr3bp_dynamics as halo_dynamics, linearized_dynamics


def compute_stm(t_span, x0):
    """
    Compute the State Transition Matrix (STM) Phi(t, t0) along a reference trajectory.
    """
    # State vector for integration: [x(6), Phi(36)]
    phi0 = np.eye(6).flatten()
    y0 = np.concatenate([x0, phi0])

    def variational_eq(t, y):
        x = y[:6]
        phi = y[6:].reshape((6, 6))

        dx = halo_dynamics(t, x)
        A = linearized_dynamics(x)
        dphi = A @ phi

        return np.concatenate([dx, dphi.flatten()])

    sol = solve_ivp(variational_eq, t_span, y0, method="DOP853", rtol=1e-9, atol=1e-12)

    phi_final = sol.y[6:, -1].reshape((6, 6))
    return phi_final


def find_natural_formation(chief_initial, period):
    """
    Find initial relative states that result in bounded/periodic relative motion.
    These correspond to the center subspace or unity eigenvalues of the Monodromy matrix.
    """
    print("Computing Monodromy Matrix...")
    monodromy = compute_stm((0, period), chief_initial)

    vals, vecs = eig(monodromy)

    print("Eigenvalues:")
    for v in vals:
        print(f"{v:.4f} (Mag: {abs(v):.4f})")

    # Select eigenvectors with magnitude close to 1 (Center/Unity)
    # Usually indices 2,3,4,5 (0,1 are stable/unstable)
    # But order is not guaranteed.

    natural_offsets = []

    for i in range(6):
        if 0.99 < abs(vals[i]) < 1.01:
            # This is a candidate for bounded motion
            # Take the real part (if complex conjugate pair, real part is a valid direction?)
            # Actually, for complex pair v, v_conj, the vectors are u, u_conj.
            # Real solutions are Re(u) and Im(u).
            vec = vecs[:, i]
            if np.iscomplex(vals[i]):
                natural_offsets.append(np.real(vec))
                natural_offsets.append(np.imag(vec))
            else:
                natural_offsets.append(np.real(vec))

    # Remove duplicates (roughly)
    unique_offsets = []
    for off in natural_offsets:
        is_new = True
        for u in unique_offsets:
            if np.linalg.norm(off - u) < 1e-3 or np.linalg.norm(off + u) < 1e-3:
                is_new = False
                break
        if is_new and np.linalg.norm(off) > 1e-6:
            unique_offsets.append(off / np.linalg.norm(off))  # Normalize

    print(f"Found {len(unique_offsets)} natural relative motion directions.")
    return unique_offsets


if __name__ == "__main__":
    # Test
    T_halo = 3.0  # Approx period
    offsets = find_natural_formation(np.array(L1_HALO_INITIAL_STATE), T_halo)
    print("Natural Offsets (Normalized):")
    for off in offsets:
        print(off)
