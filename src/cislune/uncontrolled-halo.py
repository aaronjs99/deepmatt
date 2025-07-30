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

from scipy.special import legendre
import matplotlib.pyplot as plt

from utils import *
from params import *

def haloAccel(t, p, v, *args, **kwargs):
  gamma = kwargs['gamma']
  L_type = kwargs['L_type']
  mu = kwargs['mu']
  linearity = False
  richardson = True

  c2  = cn(2, gamma, L_type, mu)
  if "m" in kwargs:
    m = kwargs["m"]
    rho =  m.sqrt(p.x**2 + p.y**2 + p.z**2)
  else:
    rho = np.sqrt(p.x**2 + p.y**2 + p.z**2)

  polyp  = [legendre(n) for n in range(3, 12)]
  polyd  = [polyp[i].deriv()                            for i in range(len(polyp))]
  Pnlist = [eval_poly1d(polyp[i].coefficients, p.x/rho) for i in range(len(polyp))]
  Pdlist = [eval_poly1d(polyd[i].coefficients, p.x/rho) for i in range(len(polyd))]
  
  aX_lin =  2*v.y + ( 2*c2 + 1)*p.x
  aY_lin = -2*v.x + (-  c2 + 1)*p.y
  aZ_lin =           -  c2     *p.z

  if linearity:
    aX_nonlin = 0.0
    aY_nonlin = 0.0
    aZ_nonlin = 0.0
  else:
    if richardson:
      aX_nonlin =  1.5*cn(3, gamma, L_type, mu)*    (2*(p.x**2) -   (p.y**2) -   (p.z**2)) + \
                   2.0*cn(4, gamma, L_type, mu)*p.x*(2*(p.x**2) - 3*(p.y**2) - 3*(p.z**2))
      aY_nonlin = -3.0*cn(3, gamma, L_type, mu)*       (p.x*p.y) + \
                  -1.5*cn(4, gamma, L_type, mu)*p.y*(4*(p.x**2) -   (p.y**2) -   (p.z**2))
      aZ_nonlin = -3.0*cn(3, gamma, L_type, mu)*       (p.x*p.z) + \
                  -1.5*cn(4, gamma, L_type, mu)*p.z*(4*(p.x**2) -   (p.y**2) -   (p.z**2))
    else:
      aX_nonlin = np.sum([cn(n, gamma, L_type, mu)*(n*rho**(n-1)*(p.x/rho)*Pnlist[n-3] +
                                                      rho**(n)*((1/rho) - ((p.x*p.x)/(rho**3)))*Pdlist[n-3]) for n in range(3, len(polyp)+3)])
      aY_nonlin = np.sum([cn(n, gamma, L_type, mu)*(n*rho**(n-1)*(p.y/rho)*Pnlist[n-3] +
                                                      rho**(n)*(        - ((p.x*p.y)/(rho**3)))*Pdlist[n-3]) for n in range(3, len(polyp)+3)])
      aZ_nonlin = np.sum([cn(n, gamma, L_type, mu)*(n*rho**(n-1)*(p.z/rho)*Pnlist[n-3] +
                                                      rho**(n)*(        - ((p.x*p.z)/(rho**3)))*Pdlist[n-3]) for n in range(3, len(polyp)+3)])
  
  return [aX_lin + aX_nonlin, aY_lin + aY_nonlin, aZ_lin + aZ_nonlin]

M_E   = 5.97219e+24
M_S   = 1.98847e+30
mu_ES = M_E/(M_E + M_S)
gamma = 1.001090475e-2

l1    = -15.965031400
l2    =   1.7409008000
Delta =   0.29221444425

c2  = cn(2, gamma, 1, mu_ES)

wp = np.sqrt((2-c2+np.sqrt(9*c2*c2-8*c2))/2)
wv = np.sqrt(c2)
Lambda = np.sqrt((c2-2+np.sqrt(9*c2*c2-8*c2))/2)
kappa  = (wp*wp + 1 + 2*c2)/(2*wp)

# Ax = 0.100
# Az = 0.100
Ax = 0.1*206000000/110000000
Az = 0.1*110000000/110000000
Ay = kappa*Ax

def runHalo(timeRange):
  pos = p_0.copy()
  vel = v_0.copy()
  list_of_pos = []
  list_of_vel = []

  for time_step in range(num_steps):
    time = t_0 + time_step * dt
    acc = haloAccel(time, pos, vel, gamma = gamma, L_type = 1, mu = mu_ES)
    
    vel.x += acc[0] * dt
    vel.y += acc[1] * dt
    vel.z += acc[2] * dt

    pos.x += vel.x * dt
    pos.y += vel.y * dt
    pos.z += vel.z * dt

    list_of_pos.append(Vector(pos.x, pos.y, pos.z))
    list_of_vel.append(Vector(vel.x, vel.y, vel.z))

  fig = plt.figure()
  ax  = fig.add_subplot(111, projection='3d')
  ax.plot([p.x for p in list_of_pos], [p.y for p in list_of_pos], [p.z for p in list_of_pos])
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  plt.show()

  fig = plt.figure()
  ax  = fig.add_subplot(111)
  ax.plot(timeRange, [v.x for v in list_of_vel], label = 'X')
  ax.plot(timeRange, [v.y for v in list_of_vel], label = 'Y')
  ax.plot(timeRange, [v.z for v in list_of_vel], label = 'Z')
  ax.set_xlabel('Time')
  ax.set_ylabel('Velocity')
  ax.legend()
  plt.show()

def optimizeHalo(initPos, initVel, timeRange, *args, **kwargs):
  m = GEKKO(remote=True, server = 'https://apmonitor.com')
  m.time = timeRange - timeRange[0]
  
  pos  = [m.Var() for i in range(3)]
  vel  = [m.Var(fixed_initial=False) for i in range(3)]
  posd = [m.Var() for i in range(3)]
  veld = [m.Var(fixed_initial=False) for i in range(3)]
  time = m.Param(value = timeRange)

  for i in range(3):
    pos[i].value = [initPos.x, initPos.y, initPos.z][i]
    vel[i].value = [initVel.x, initVel.y, initVel.z][i]

  for i in range(3):
    m.delay(pos[i], posd[i], periodicity_time)
    m.delay(vel[i], veld[i], periodicity_time)
    m.Equation(pos[i].dt() == vel[i])
    m.Equation(vel[i].dt() == haloAccel(time, Vector(pos[0], pos[1], pos[2]),
                                              Vector(vel[0], vel[1], vel[2]),
                                        gamma = gamma, L_type = 1, mu = mu_ES, m = m)[i])

  m.options.IMODE = 9
  m.options.MAX_ITER = 3000
  m.options.SOLVER = 3
  try:
    print("Solving...")
    m.solve(disp = True)
  except:
    print("Failed to solve")
    m.cleanup()
    del m
    return [[] for j in range(3)], [[] for j in range(3)], [[] for j in range(3)], [[] for j in range(3)]

  pos  = [pos [i].value for i in range(3)]
  vel  = [vel [i].value for i in range(3)]
  posd = [posd[i].value for i in range(3)]
  veld = [veld[i].value for i in range(3)]
  # posd = [pos [i].value for i in range(3)]
  # veld = [vel [i].value for i in range(3)]

  m.cleanup()
  del m
  
  print("Initial Position : ", [pos[0][ 0], pos[1][ 0], pos[2][ 0]])
  print("Final   Position : ", [pos[0][-1], pos[1][-1], pos[2][-1]])
  
  return pos, vel, posd, veld

def iterateHalo(p_0, v_0, t_0, dt, num_steps, periodicity_time, num_iterations, iteration_time):
  allTime = np.linspace(0, num_iterations*iteration_time, num_iterations*num_steps)
  allPos  = [[] for i in range(3)]
  allVel  = [[] for i in range(3)]
  allPosd = [[] for i in range(3)]
  allVeld = [[] for i in range(3)]
  
  initPos = p_0.copy()
  initVel = v_0.copy()
  prevData = [[0 for i in range(num_steps)] for j in range(3)]
  for i in range(num_iterations):
    print(f"Running iteration  {i+1} of {num_iterations}")
    timeRange = np.linspace(i*iteration_time, (i+1)*iteration_time, num_steps)
    if i == 0:
      pos, vel, posd, veld = optimizeHalo(initPos, initVel, timeRange)
    else:
      for j in range(3):
        prevData[j][0:periodicity_time] = allPos[j][-periodicity_time:]
      pos, vel, posd, veld = optimizeHalo(initPos, initVel, timeRange, prevData = prevData)
    for j in range(3):
      allPos [j].extend(pos [j])
      allVel [j].extend(vel [j])
      allPosd[j].extend(posd[j])
      allVeld[j].extend(veld[j])
    initPos = Vector(pos[0][-1], pos[1][-1], pos[2][-1])
    initVel = Vector(vel[0][-1], vel[1][-1], vel[2][-1])

  return allTime, allPos, allVel, allPosd, allVeld

def plotter(allTime, allPos, allVel, allPosd, allVeld, fig = None):
  fig = plt.figure()
  ax  = fig.add_subplot(111, projection='3d')
  ax.plot(allPos[0], allPos[1], allPos[2])
  ax.set_title('Optimized Halo Orbit')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  plt.show()

  fig = plt.figure()
  ax  = fig.add_subplot(111)
  ax.plot(allTime, [(allPos[0][i]-allPosd[0][i])**2 + (allPos[1][i]-allPosd[1][i])**2 for i in range(len(allTime))], label = 'Circular Error')
  ax.plot(allTime, [(allPos[2][i]-allPosd[2][i])**2 for i in range(len(allTime))], label = 'Z Error')
  ax.set_title('Error')
  ax.set_xlabel('Time')
  ax.set_ylabel('Error')
  plt.show()
  
  fig = plt.figure()
  ax  = fig.add_subplot(111)
  for i in range(3):
    ax.plot(allTime, allPos [i], label = f'Position {i}')
    # ax.plot(allTime, allPosd[i], label = f'Delayed Position {i}')
  ax.set_title('Position')
  ax.set_xlabel('Time')
  ax.set_ylabel('Position')
  ax.legend()
  plt.show()

  fig = plt.figure()
  ax  = fig.add_subplot(111)
  for i in range(3):
    ax.plot(allTime, allVel[i], label = f'Velocity {i}')
  ax.set_title('Velocity')
  ax.set_xlabel('Time')
  ax.set_ylabel('Velocity')
  ax.legend()
  plt.show()

if __name__ == "__main__":
  p_0 =  Vector(-Ax,     0, -Az)
  V_0 = []
  for i in range(1):
    # V_0.append(Vector(0.05*i*Ay*wp, np.sqrt(1.00-(0.05*i)**2)*Ay*wp, 0.0))
    V_0.append(Vector(np.sqrt(1.00-(0.05*i)**2)*Ay*wp, 0.05*i*Ay*wp, 0.0))
    # V_0.append(Vector(0.0, 0.0, 0.0))
    # V_0.append(Vector(Ay*wp, 0.0, 0.0))
  
  t_0   = 0.000
  dt    = 0.01*np.pi
  num_steps        = 150
  periodicity_time =  50
  num_iterations   =   1
  iteration_time   = num_steps*dt

  allTime = np.linspace(0, num_iterations*iteration_time, num_iterations*num_steps)
  AllPos  = []
  AllVel  = []
  AllPosd = []
  AllVeld = []
  
  for iter in range(len(V_0)):
    print(f"Running halo orbit {iter+1} of {len(V_0)}")
    v_0 = V_0[iter]
    allTime, allPos, allVel, allPosd, allVeld = iterateHalo(p_0, v_0, t_0, dt, num_steps, periodicity_time, num_iterations, iteration_time)
    AllPos .append(allPos )
    AllVel .append(allVel )
    AllPosd.append(allPosd)
    AllVeld.append(allVeld)
  
  fig = plt.figure()
  ax  = fig.add_subplot(111, projection='3d')
  for i in range(len(AllPos)):
    ax.plot(AllPos[i][0], AllPos[i][1], AllPos[i][2], color = (0.0, 0.0, 1.0))
  ax.set_title('Optimized Halo Orbit')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  plt.show()

  plotter(allTime, AllPos[0], AllVel[0], AllPosd[0], AllVeld[0])