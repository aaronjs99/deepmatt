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

from sympy import symbols, Eq, solve

from utils import *

## ACTUAL VALUES
# mu_M = 4.9048695900e+12
# mu_E = 3.9860044188e+14
# dist_EM = 3.84399000e+8

## SIMULATED VALUES
mu_M = 1.0
# mu_E = 81.2662670365
# mu_E = 333030.0
mu_E = 1.0
dist_EM = 1.0

mu = mu_M/(mu_E + mu_M)
p_E = Vector( -mu, 0.0, 0.0)
p_M = Vector(1-mu, 0.0, 0.0)

Isp = 1.0
g_0 = 1.0

x = symbols('x')
eqL1 = Eq(x**5 + ( (mu-3))*x**4 + (3-2*mu)*x**3 + (-mu)*x**2 + ( (2*mu))*x + (-mu), 0)
eqL2 = Eq(x**5 + (-(mu-3))*x**4 + (3-2*mu)*x**3 + (-mu)*x**2 + (-(2*mu))*x + (-mu), 0)
L1 = (1-mu) - solve(eqL1, dict=True)[0][x] 
L2 = (1-mu) + solve(eqL2, dict=True)[0][x] 

if __name__ == "__main__":
  print(L1)
  print(L2)