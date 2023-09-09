
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import os
from PIL import Image
import matplotlib.animation as animation
from scipy.fftpack import fftn, ifftn
from turbulent_interp import generate_turbulent_field as gtf
from scipy.interpolate import RegularGridInterpolator

# Constants in model
g = 9.792

# Constants
m = 0.733 # max (starting) mass
mass_min = 0.200

# Mass function
def m(t):
    if mass_min + (0.532 * (1.5 - t)) / 1.5 < mass_min:
        return mass_min
    return mass_min + (0.532 * (1.5 - t)) / 1.5

# Thrust function
def thrust(t, t_ignition, t_cutoff, max_thrust):
    if t < t_ignition:
        return max_thrust * (t / t_ignition)
    elif t_ignition <= t <= t_cutoff:
        return max_thrust
    elif t > t_cutoff:
        return 0

# Define parabolic Thrust Curve Function
def parabolic_thrust(x,max_thrust):
    scaling = np.sqrt(max_thrust)/0.75
    parabolic_bracket = (x*scaling - np.sqrt(max_thrust))
    return max(-(parabolic_bracket*parabolic_bracket) + max_thrust,0)

# Constants for thrust function
t_ignition = 0.5
t_cutoff = 10
max_thrust = 250


rho = 1.22264 # air density (kg/m^3) 
CD = 0.697 # coefficient of drag 
A = 0.00456 # cross-sectional area (m^2) 

# Launch angles
theta = np.radians(45) # angle between thrust and the vertical axis in the x-z plane
phi = np.radians(90) # angle between thrust and the vertical axis in the x-y plane

# Constants
CD_x = 0.697  # coefficient of drag in the x-direction
A_x = 0.00456 # cross-sectional area in the x-direction (m^2)

CD_y = 0.697  # coefficient of drag in the y-direction
A_y = 0.00456 # cross-sectional area in the y-direction (m^2)

CD_z = 0.697  # coefficient of drag in the z-direction
A_z = 0.00456 # cross-sectional area in the z-direction (m^2)

tx_ls = []

# Generate a turbulent wind field
N = 200
w_x = gtf(N, seed=0)
w_y = gtf(N, seed=1)
w_z = gtf(N, seed=2)

# Create an interpolating function for the wind field
x = np.linspace(0, N-1, N)
y = np.linspace(0, N-1, N)
z = np.linspace(0, N-1, N)
# Create interpolating functions for each component
interpolator_x = RegularGridInterpolator((x, y, z), w_x)
interpolator_y = RegularGridInterpolator((x, y, z), w_y)
interpolator_z = RegularGridInterpolator((x, y, z), w_z)
# Set the max values for x, y, and z
x_max = 6
y_max = 600
z_max = 400
# ODE function
def f(t, v):
    # Normalize the physical coordinates to the grid space
    x_grid = max((v[0] / x_max) * (N - 1), 0)
    y_grid = max((v[1] / y_max) * (N - 1), 0)
    z_grid = max((v[2] / z_max) * (N - 1), 0)
    #print("x_grid:", x_grid)
    #print("y_grid:", y_grid)
    #print("z_grid:", z_grid)

    # Create the grid point
    point = np.array([x_grid, y_grid, z_grid])
    #print("point:", point)
    # In the ODE function, interpolate each component separately
    w_interp_x = interpolator_x(point)[0]
    w_interp_y = interpolator_y(point)[0]
    w_interp_z = interpolator_z(point)[0]
    #print("w_interp_x:", w_interp_x)
    #print("w_interp_y:", w_interp_y)
    #print("w_interp_z:", w_interp_z)
    #  Compute the relative velocity
    v_rel = v[3:] - np.array([w_interp_x, w_interp_y, w_interp_z])
    #print(v[3:])
    #tx_ls.append(v[:3])
    D_x = -(1/2) * rho * CD_x * A_x * v_rel[0]**2
    D_y = -(1/2) * rho * CD_y * A_y * v_rel[1]**2
    D_z = -(1/2) * rho * CD_z * A_z * v_rel[2]**2

    if v_rel[0] < 0:
        D_x = -1 * D_x
    if v_rel[1] < 0:
        D_y = -1 * D_y
    if v_rel[2] < 0:
        D_z = -1 * D_z

    T_x = parabolic_thrust(t,max_thrust) * np.cos(theta) * np.cos(phi)
    T_y = parabolic_thrust(t,max_thrust) * np.cos(theta) * np.sin(phi)
    T_z = parabolic_thrust(t,max_thrust) * np.sin(theta)

    return np.array([v[3],\
                     v[4],\
                     v[5],\
                     (1/m(t)) * (T_x + D_x),\
                     (1/m(t)) * (T_y + D_y),\
                     (1/m(t)) * (T_z - m(t) * g + D_z)])
# Initial conditions
yinit = np.array([0, 0, 0, 0, 0, 0])

t_max = 20
t = np.linspace(0, t_max, t_max * 100)

# Solve ODE
y = solve_ivp(f, [0, t_max], yinit, t_eval=t, method='RK45')

# Plot the solutions in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(y.y[0], y.y[1], y.y[2], label='Trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Animation function
#def animate(i):
#    ax.view_init(elev=30., azim=3.6 * i)

# Create an animation object
#ani = animation.FuncAnimation(fig, animate, frames=100, interval=50)

# Save the animation as a GIF
#ani.save("3d_turbulent_animation.gif", writer="imagemagick", fps=15)


plt.show()


