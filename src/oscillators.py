# /***********************************************************
# *                                                         *
# * Copyright (c) 2024                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu, Brett T. Lopez                *
# * Contact: {aaronjs, btlopez}@ucla.edu                    *
# *                                                         *
# ***********************************************************/

from commons import *
from params import *
import matplotlib.pyplot as plt

def runVanDerPol():
  mu = 0.4
  m = GEKKO(remote=True, server='https://apmonitor.com') # or 'https://byu.apmonitor.com'
  m.time = np.linspace(0, 80, 1000)

  x1 = m.Var(value=0.1)
  x2 = m.Var(value=0.1)
  m.Equation(x1.dt() == x2)
  m.Equation(x2.dt() == mu*(1-x1**2)*x2 - x1)

  m.options.IMODE = 4
  m.solve(disp=True)

  plt.plot(m.time, x1, label='x1')
  plt.plot(m.time, x2, label='x2')
  plt.legend()
  plt.show()

def runDuffing():
  alpha = 1.00
  beta  = 0.40
  gamma = 0.30
  delta = 0.02
  omega = 1.20

  dt = 0.1
  totTime  = 30
  numIterations = 5

  initPos = [0.1, 0.1]
  X = [[] for j in range(2)]

  for i in range(numIterations):
    m = GEKKO(remote=True)
    m.time = np.linspace(i*totTime, (i+1)*totTime, int(totTime/dt))

    t  = m.Param(value=m.time)
    x1 = m.Var(value=initPos[0])
    x2 = m.Var(value=initPos[1])
    m.Equation(x1.dt() == x2)
    m.Equation(x2.dt() == -delta*x2 - alpha*x1 -beta*(x1**3) + gamma*m.cos(omega*t))    # MODIFY

    m.options.IMODE = 4
    m.options.SOLVER = 2
    m.solve(disp=True)

    X[0].extend([x1[k] for k in range(len(x1))])
    X[1].extend([x2[k] for k in range(len(x2))])
    initPos = [x1[-1], x2[-1]]
    
    m.cleanup()
    del m

  time = np.linspace(0, totTime*numIterations, int(totTime*numIterations/dt))

  fig = plt.figure()
  ax  = fig.add_subplot(111)
  ax.plot(time, X[0], label='x1')
  ax.plot(time, X[1], label='x2')
  ax.legend()
  plt.show()

  fig = plt.figure()
  ax  = fig.add_subplot(111)
  ax.plot(X[0], X[1])
  plt.show()

def runCoupledVanDerPol():
  numSteps = 0.1
  totTime  = 20
  numIterations = 7
  a       = 0.5
  epsilon = 0.4
  gamma   = 0.01
  
  A =  [[ 0.0,  0.0,  0.0,               1.0,               0.0,               0.0,],
        [ 0.0,  0.0,  0.0,               0.0,               1.0,               0.0,],
        [ 0.0,  0.0,  0.0,               0.0,               0.0,               1.0,],
        [-1.0,  0.0,  0.0, epsilon*(2-gamma),               0.0, epsilon*   gamma ,],
        [ 0.0, -1.0,  0.0, epsilon*   gamma , epsilon*(2-gamma),               0.0,],
        [ 0.0,  0.0, -1.0,               0.0, epsilon*   gamma , epsilon*(2-gamma),],]

  initPos = [10.0, 0.0, 0.0, 0.01, 0.001, 0.0001]
  # initPos = [4.1618861492, 1.1257495892, 0.074600534433, -0.51991868544,  0.79438678913,  0.20863691021]
  # initPos = [1.1878766714,  1.3543417206, 2.203626175700,  2.82290784170, -1.37974443720, -0.44834495344]
  # initPos = [2.0864189516, -1.6214927740, 0.497639832590, -0.67534346163, -2.37514316410, -2.15997049650]
  X = [[] for j in range(6)]

  for i in range(numIterations):
    m = GEKKO(remote=True)
    m.time = np.linspace(0, totTime, int(totTime/numSteps))
    x = [m.Var(value=initPos[j]) for j in range(6)]
    for j in range(3):
      m.Equation(x[j].dt() == x[j+3])
      m.Equation(x[j+3].dt() == np.dot(A[j+3], [x[k] for k in range(6)]) - a*x[j]*x[j]*x[j+3])

    m.options.IMODE = 4
    m.options.SOLVER = 2
    m.options.MAX_ITER = 1000
    try:
      m.solve(disp=True)
    except:
      print("Failed to solve")
      break

    for j in range(6):
      X[j].extend([x[j][k] for k in range(len(x[j]))])
    initPos = [x[j][-1] for j in range(6)]
    m.cleanup()
    del m

  time = np.linspace(0, totTime*numIterations, int(totTime*numIterations/numSteps))

  fig = plt.figure()
  ax  = fig.add_subplot(111, projection='3d')
  ax.plot(X[0], X[1], X[2])
  plt.show()

  fig = plt.figure()
  ax  = fig.add_subplot(111)
  ax.plot(time, X[0], label='p_X')
  ax.plot(time, X[1], label='p_Y')
  ax.plot(time, X[2], label='p_Z')
  # ax.plot(time, X[3], label='v_X')
  # ax.plot(time, X[4], label='v_Y')
  # ax.plot(time, X[5], label='v_Z')
  ax.legend()
  plt.show()

if __name__ == "__main__":
  runVanDerPol()
  # runCoupledVanDerPol()
  # runDuffing()