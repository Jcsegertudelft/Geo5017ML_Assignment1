import numpy as np
import matplotlib.pyplot as plt


# Tracked Positions of the drone (x, y, z) and the corresponding time stamps (t)
pos = np.array([[2,0,1],[1.08,1.68,2.38],[-0.83,1.82,2.49],
               [-1.97,0.28,2.15],[-1.31,-1.51,2.59],[0.57,-1.91,4.32]])
t = np.array([1,2,3,4,5,6])

# Extract the x, y, z coordinates and time stamps
x = pos[:,0]
y = pos[:,1]
z = pos[:,2]

# Create the 3D figure with the tracked positions and the trajectory
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z, color='navy', label='Tracked Positions')
ax.plot(x,y,z, linewidth=1.4, color='crimson', label='Trajectory')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Drone Trajectory', pad=20)
ax.text(x[0]+ 0.2,y[0],z[0],'t = 1')
ax.text(x[-1]+ 0.2,y[-1],z[-1],'t = 6')
ax.legend()
plt.tight_layout()
plt.savefig('Drone_Trajectory.png', dpi = 300)
plt.show()
