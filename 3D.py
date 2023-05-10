
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import os
from PIL import Image
import matplotlib.animation as animation

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
# ODE function
def f(t, v):
    D_x = -(1/2) * rho * CD_x * A_x * v[3] * v[3]
    D_y = -(1/2) * rho * CD_y * A_y * v[4] * v[4]
    D_z = -(1/2) * rho * CD_z * A_z * v[5] * v[5]

    if v[3] < 0:
        D_x = -1 * D_x
    if v[4] < 0:
        D_y = -1 * D_y
    if v[5] < 0:
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
def animate(i):
    ax.view_init(elev=30., azim=3.6 * i)

# Create an animation object
ani = animation.FuncAnimation(fig, animate, frames=100, interval=50)

# Save the animation as a GIF
ani.save("3d_plot_animation.gif", writer="imagemagick", fps=15)






plt.show()

