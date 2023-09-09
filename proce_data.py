

# import libraries
import pandas as pd
import numpy as np
import matplotlib
# Set the backend to TkAgg
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


# read in data
df = pd.read_csv('All_Data.csv')

# get 'Altitude (ft)' column, 'Lateral distance (ft)' column and 'Time (s)' column out save as new dataframe
df_alt = df[['Altitude (ft)', 'Lateral distance (ft)', 'Time (s)']]

# remove all rows with NaN values in 'Altitude (ft)' column
df_alt = df_alt.dropna(subset=['Altitude (ft)'])

# Define the update function for the animation
def update_graph(num):
    ax.view_init(elev=10., azim=num)
    return ax

# plot a 3D scatter graph of the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_alt['Lateral distance (ft)'], df_alt['Time (s)'], df_alt['Altitude (ft)'], c='r', marker='o')
ax.set_xlabel('Lateral distance (ft)')
ax.set_ylabel('Time (s)')
ax.set_ylim3d(0, 600)
ax.set_zlabel('Altitude (ft)')
ax.set_yticks(ax.get_yticks()[::160])
plt.show()

# Create the animation
ani = animation.FuncAnimation(fig, update_graph, frames=np.arange(0, 360, 1), interval=50, blit=False, repeat=True)


ani.save('3d_scatter_animation.gif', writer='pillow', fps=15)
plt.show()






#print(df_alt.tail(10))




