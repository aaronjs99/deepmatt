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

from gekko import GEKKO
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import *
from params import *

mu = 0.012150585609741648

def hausdorff_distance(A, B):
  """
  Compute the Hausdorff distance between two sets of points A and B.
  """
  dist = 0
  for a in A:
    min_dist = np.inf
    for b in B:
      dist = np.linalg.norm(a - b)
      if dist < min_dist:
        min_dist = dist
    if min_dist > dist:
      dist = min_dist
  return dist

def import_patch_points(file):
  points = []
  with open(file, "r") as f:
    for line in f:
      points.append(np.array([float(x) for x in line.split(sep=',')]))
  return points

def f_c(theta, r, u, epsilon, *args, **kwargs):
  m = kwargs["m"]
  factor = 1.0/(1.0 + epsilon*m.cos(theta))
  r1 = m.sqrt((r[0] +    mu )**2 + r[1]**2 + r[2]**2)
  r2 = m.sqrt((r[0] - (1-mu))**2 + r[1]**2 + r[2]**2)
  ax =  2*r[4] + factor*(r[0] - (1-mu)*(r[0] + mu)/r1**3 - mu*(r[0] - (1-mu))/r2**3)
  ay = -2*r[3] + factor*(r[1] - (1-mu)* r[1]      /r1**3 - mu* r[1]          /r2**3)
  az =    r[2] + factor*(     - (1-mu)* r[2]      /r1**3 - mu* r[2]          /r2**3)
  return [ax, ay, az]

def stationkeeping(p_0, v_0, u_max):
  halo_orbit = import_patch_points("halo_orbit_points.csv")
  robot_position = p_0
  robot_velocity = v_0
  dt = 0.01
  T = 1.5

  control_history = []
  position_history = []
  velocity_history = []
  error_history = []
  control_input_history = []
  time_history = []

  m = GEKKO(remote=True, server='http://apmonitor.com')
  m.time = np.linspace(0, T, int(T/dt))

  theta = m.Var(value = 0.0)
  r     = m.Array(m.Var, 6)
  u     = m.Array(m.Var, 3)

  for i in range(3):
    r[i+0].value = robot_position[i]
    r[i+3].value = robot_velocity[i]
    u[i+0].value = 0

  m.Equation(theta.dt() == 1.0)
  for i in range(3):
    m.Equation(r[i+0].dt() == r[i+3])
    m.Equation(r[i+3].dt() == f_c(theta, r, u, 0, m = m)[i+0])

  # for i in range(3):
    # m.Equation(u[i] <=  u_max)
    # m.Equation(u[i] >= -u_max)

  final      = m.Param(value = np.zeros(len(m.time)))
  integrator = m.Param(value = np.ones (len(m.time)))
  final.value[-1] = 1.0
  
  # calculate minimum of distance between robot position and patch points
  dist_to_patch_points = [m.sqrt((r[0] - p[0])**2 + (r[1] - p[1])**2 + (r[2] - p[2])**2) for p in halo_orbit[::5000]]
  print(len(dist_to_patch_points))
  min_dist_to_patch_points = dist_to_patch_points[0]
  for i in range(1, len(dist_to_patch_points)):
    min_dist_to_patch_points = m.min3(min_dist_to_patch_points, dist_to_patch_points[i])
  m.Minimize(integrator*(0.1 * integrator * min_dist_to_patch_points +
                         0.0 * (u[0]**2 + u[1]**2 + u[2]**2)))

  m.options.SOLVER = 2
  m.options.IMODE = 6
  m.solve(disp=True)

  for i in range(len(m.time)):
    position_history.append([r[0][i], r[1][i], r[2][i]])
    velocity_history.append([r[3][i], r[4][i], r[5][i]])
    control_input_history.append([u[0][i], u[1][i], u[2][i]])
    time_history.append(m.time[i])

  return position_history, velocity_history, control_input_history, time_history

  
if __name__ == "__main__":
  p_0 = np.array([8.2339090555970500e-01,
                  0.0000000000000000e+00,
                  1.1103208330134800e-06,])
  v_0 = np.array([0.0000000000000000e+00,
                  1.2632639894667600e-01,
                  0.0000000000000000e+00,])
  position_history, velocity_history, control_input_history, time_history = stationkeeping(p_0, v_0, u_max=1000)
  # print("Position History: ", position_history)
  # print("Velocity History: ", velocity_history)
  # print("Control Input History: ", control_input_history)
  # print("Time History: ", time_history)
  # print("Hausdorff Distance: ", hausdorff_distance(import_patch_points("halo_orbit.txt"), position_history))
  # plot position
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot([p[0] for p in position_history], [p[1] for p in position_history], [p[2] for p in position_history])
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_title('Stationkeeping')
  plt.show()

  # plot control input
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(time_history, [u[0] for u in control_input_history], label='u_x')
  ax.plot(time_history, [u[1] for u in control_input_history], label='u_y')
  ax.plot(time_history, [u[2] for u in control_input_history], label='u_z')
  ax.set_xlabel('Time')
  ax.set_ylabel('Control Input')
  ax.set_title('Stationkeeping')
  ax.legend()
  plt.show()