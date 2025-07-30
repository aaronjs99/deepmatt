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

import numpy as np
from gekko import GEKKO
import scipy.optimize as opt
import matplotlib.pyplot as plt    

def chase(initPos, k1, k2, bounds):
  m = GEKKO(remote=True, server='https://apmonitor.com')
  m.time = np.linspace(0, 1, 201)
  
  # State and Input Variables
  x_T = m.Var(value=initPos[0][0], fixed_initial=True)
  v_T = m.Var(value=initPos[0][1], fixed_initial=True)
  a_T = m.Var(value=initPos[0][2], fixed_initial=True)
  x_C = m.Var(value=initPos[1][0], fixed_initial=True)
  v_C = m.Var(value=initPos[1][1], fixed_initial=True)
  a_C = m.Var(value=initPos[1][2], fixed_initial=True)
  u_T = m.Var()
  u_C = m.Var(value=0, fixed_initial=False)
  # Time Variable
  T = m.FV(1, lb=1e-10, ub=1e10)
  T.STATUS = 1

  # Time-based Parameters
  integrator = m.Param(value=np. ones_like(m.time))
  final      = m.Param(value=np.zeros_like(m.time))
  final[-1] = 1.0
  
  # State Dynamics
  m.Equation(x_T.dt()/T == v_T)
  m.Equation(v_T.dt()/T == a_T)
  m.Equation(a_T.dt()/T == u_T)
  m.Equation(x_C.dt()/T == v_C)
  m.Equation(v_C.dt()/T == a_C)
  m.Equation(a_C.dt()/T == u_C)
  # Input Definition and Bounds
  m.Equation(u_T == bounds[0][1]*m.tanh(100*(x_T - x_C)))
  m.Equation(u_C >= bounds[1][0])
  m.Equation(u_C <= bounds[1][1])
  # Final State Constraints
  m.Equation((x_T - x_C)*final < 0.001)

  # Objective
  m.Minimize(final*(k1*T) + 
             integrator*(k2*(u_T**2)))
  # Solver Options
  m.options.IMODE = 6
  m.options.SOLVER = 3
  m.options.MAX_ITER = 1000
  try:
    m.solve(disp=False)
  except:
    return

  # Plotting
  if False:
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(m.time*T.value, x_T.value, 'k-' , label='Target')
    plt.plot(m.time*T.value, x_C.value, 'r--', label='Chaser')
    plt.ylabel('Position')
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(m.time*T.value, v_T.value, 'k-' , label='Target')
    plt.plot(m.time*T.value, v_C.value, 'r--', label='Chaser')
    plt.ylabel('Velocity')
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(m.time*T.value, a_T.value, 'k-' , label='Target')
    plt.plot(m.time*T.value, a_C.value, 'r--', label='Chaser')
    plt.ylabel('Acceleratiom')
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(m.time*T.value, u_T.value, 'k-' , label='Target')
    plt.plot(m.time*T.value, u_C.value, 'r--', label='Chaser')
    plt.ylabel('Control')
    plt.legend()
    plt.xlabel('Time')
    plt.show()
  
  return (T.value[-1], np.sum([u**2 for u in u_C.value]))

if __name__ == "__main__":
  initPos = [(1.00, 0.50, 0.50),
             (0.00, 0.00, 0.00)]
  bounds  = [(-1.00, 1.00),
             (-2.00, 2.00)]
  
  pareto = []

  k1 = 1.0
  for k2 in np.linspace(0.0, 1.0, 11):
    result = chase(initPos, k1, k2, bounds)
    if result is not None:
      pareto.append(result)
      print(f'k1 = {k1}, k2 = {k2}, Time = {pareto[-1][0]}, Control Effort = {pareto[-1][1]}')
    else:
      print(f'k1 = {k1}, k2 = {k2}, Failed to solve the optimization problem')
  
  # Plot the Pareto Front
  pareto = np.array(pareto)
  plt.figure()
  plt.plot(pareto[:, 0], pareto[:, 1], 'k-')
  plt.xlabel('Time')
  plt.ylabel('Control Effort')
  plt.show()