# /***********************************************************
# *                                                         *
# * Copyright (c) 2025                                      *
# *                                                         *
# * University of California, Los Angeles                   *
# *                                                         *
# * Author : Aaron John Sabu                                *
# * Contact: aaronjs@g.ucla.edu                             *
# *                                                         *
# ***********************************************************/

from cislune.utils import *


def runLissajous():
    # l1    = -15.9650314
    # l2    =  1.740900800
    # Delta =  0.29221444425
    # xL    =  0.9899871
    # c2    = mu/(np.abs(xL-1+mu)**3) + (1-mu)/(np.abs(xL+mu)**3)

    # wp     = np.sqrt((2-c2+np.sqrt(9*c2*c2-8*c2))/2)
    # wv     = np.sqrt(c2)
    #   wp = 2.086453455
    # wv = 2.0152105515
    wp = 2.00
    wv = wp
    # Lambda = np.sqrt((c2+np.sqrt(9*c2*c2-8*c2))/2)
    kappa = 3.2292680962
    # kappa  = (wp*wp + 1 + 2*c2)/(2*wp)
    # kappa = 2*Lambda/(Lambda*Lambda + 1 - c2)

    # Ax  = 200.0
    # Az  = np.sqrt((- Delta - l1*Ax*Ax)/l2)
    Ax = 0.1 * 206000000 / 110000000
    Az = 0.1 * 110000000 / 110000000
    Ay = kappa * Ax
    phi_x = 0.0
    phi_z = phi_x + 3 * np.pi / 2

    t = np.linspace(0, 3 * np.pi, 300)
    x = -Ax * np.cos(wp * t + phi_x)
    y = Ay * np.sin(wp * t + phi_x)
    z = Az * np.sin(wv * t + phi_z)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z)
    ax.set_xlim([-0.50, 0.50])
    ax.set_ylim([-0.50, 0.50])
    ax.set_zlim([-0.50, 0.50])
    # ax.set_xlim([-500000000, 500000000])
    # ax.set_ylim([-500000000, 500000000])
    # ax.set_zlim([-500000000, 500000000])
    ax.view_init(30, 135)
    plt.show()

    fig = plt.figure()
    # axbound = max(np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z))) * 1.05
    axbound = 0.60 * 1.05
    ax = fig.add_subplot(111)
    ax.plot(t, x, label="Position 1")
    ax.plot(t, y, label="Position 2")
    ax.plot(t, z, label="Position 3")
    ax.set_xlim([0, 3 * np.pi])
    ax.set_ylim([-axbound, axbound])
    ax.set_xlabel("Time")
    ax.set_ylabel("Position")
    ax.legend()
    plt.show()

    fig = plt.figure()
    v_x = np.gradient(x, t)
    v_y = np.gradient(y, t)
    v_z = np.gradient(z, t)
    #   axbound = max(np.max(np.abs(v_x)), np.max(np.abs(v_y)), np.max(np.abs(v_z))) * 1.05
    axbound = 1.25 * 1.05
    ax = fig.add_subplot(111)
    ax.plot(t, v_x, label="Velocity 1")
    ax.plot(t, v_y, label="Velocity 2")
    ax.plot(t, v_z, label="Velocity 3")
    ax.set_xlim([0, 3 * np.pi])
    ax.set_ylim([-axbound, axbound])
    ax.set_xlabel("Time")
    ax.set_ylabel("Velocity")
    ax.legend()
    plt.show()


def runRichardson():
    l1 = -15.965031400
    l2 = 1.7409008000
    s1 = -0.8246608317
    s2 = 0.1210985938
    # Delta =  0.29221444425
    # xL    =  0.9899871
    # c2    = mu/(np.abs(xL-1+mu)**3) + (1-mu)/(np.abs(xL+mu)**3)

    # wp     = np.sqrt((2-c2+np.sqrt(9*c2*c2-8*c2))/2)
    # wv     = np.sqrt(c2)
    wp = 2.086453455
    wv = 2.0152105515
    # Lambda = np.sqrt((c2+np.sqrt(9*c2*c2-8*c2))/2)
    kappa = 3.2292680962
    # kappa  = (wp*wp + 1 + 2*c2)/(2*wp)
    # kappa = 2*Lambda/(Lambda*Lambda + 1 - c2)

    # Ax  = 200.0
    # Az  = np.sqrt((- Delta - l1*Ax*Ax)/l2)
    Ax = 206000000
    Az = 110000000
    Ay = kappa * Ax
    phi_x = 0.4
    phi_z = phi_x + 3 * np.pi / 2

    delta_m = 1  # or -1

    # c2 = cn(2, gamma, 1, mu_sys)
    # c3 = cn(3, gamma, 1, mu_sys)
    # c4 = cn(4, gamma, 1, mu_sys)
    c2 = 4.0610735668
    c3 = 3.0200105081
    c4 = 3.0305378797

    d1 = (3 * (wp**2) / kappa) * (kappa * (6 * (wp**2) - 1) - 2 * wp)
    d2 = (8 * (wp**2) / kappa) * (kappa * (11 * (wp**2) - 1) - 2 * wp)

    a21 = 3 * c3 * (kappa**2 - 2) / (4 * (1 + 2 * c2))
    a22 = 3 * c3 / (4 * (1 + 2 * c2))
    a23 = (-3 * c3 * wp / (4 * kappa * d1)) * (
        3 * (kappa**3) * wp - 6 * (kappa**2) + 6 * kappa * wp + 4
    )
    a24 = (-3 * c3 * wp / (4 * kappa * d1)) * (2 + 3 * kappa * wp)
    b21 = (-3 * c3 * wp / (2 * d1)) * (3 * kappa * wp - 4)
    b22 = -3 * c3 * wp / d1
    d21 = -c3 / (2 * (wp**2))
    a31 = (-9 * wp / (4 * d2)) * (
        4 * c3 * (kappa * a23 - b21) + kappa * c4 * (4 + (kappa**2))
    ) + ((9 * (wp**2) + 1 - c2) / (2 * d2)) * (
        3 * c3 * (2 * a23 - kappa * b21) + c4 * (2 + 3 * (kappa**2))
    )
    a32 = (-9 * wp / (4 * d2)) * (4 * c3 * (3 * kappa * a24 - b22) + kappa * c4) - (
        (3 / (2 * d2)) * (9 * (wp**2) + 1 - c2)
    ) * (c3 * (kappa * b22 + d21 - 2 * a24) - c4)
    b31 = (3 / (8 * d2)) * (
        8 * wp * (3 * c3 * (kappa * b21 - 2 * a23) - c4 * (2 + 3 * (kappa**2)))
        + (9 * (wp**2) + 1 - c2)
        * (4 * c3 * (kappa * a23 - b21) + (kappa * c4 * (4 + (kappa**2))))
    )
    b32 = (9 * wp / d2) * (c3 * (kappa * b22 + d21 - 2 * a24) - c4) + (3 / (8 * d2)) * (
        9 * (wp**2) + 1 + 2 * c2
    ) * (4 * c3 * (kappa * a24 - b22) + kappa * c4)
    d31 = (3 / (64 * (wp**2))) * (4 * c3 * a24 + c4)
    d32 = (3 / (64 * (wp**2))) * (4 * c3 * (a23 - d21) + c4 * (4 + (kappa**2)))

    t = np.linspace(0, 10 * np.pi, 1000)
    tau = t * wp + phi_x
    x = (
        a21 * (Ax**2)
        + a22 * (Az**2)
        - Ax * np.cos(tau)
        + (a23 * (Ax**2) - a24 * (Az**2)) * np.cos(2 * tau)
        + (a31 * (Ax**3) - a32 * (Ax * (Az**2))) * np.cos(3 * tau)
    )
    y = (
        Ay * np.sin(tau)
        + (b21 * (Ax**2) - b22 * (Az**2)) * np.sin(2 * tau)
        + (b31 * (Ax**3) - b32 * (Ax * (Az**2))) * np.sin(3 * tau)
    )
    z = (
        delta_m * Az * np.cos(tau)
        + delta_m * d21 * Ax * Az * (np.cos(2 * tau) - 3)
        + delta_m * (d32 * Az * (Ax**2) - d31 * (Az**3)) * np.cos(3 * tau)
    )

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z)
    plt.show()


if __name__ == "__main__":
    # runRichardson()
    runLissajous()
