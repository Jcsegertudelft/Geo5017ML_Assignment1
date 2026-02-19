import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Tracked Positions of the drone (x, y, z) and the corresponding time stamps (t)
pos = np.array([[2,0,1],[1.08,1.68,2.38],[-0.83,1.82,2.49],
               [-1.97,0.28,2.15],[-1.31,-1.51,2.59],[0.57,-1.91,4.32]])
t = np.array([1,2,3,4,5,6])

# Reshape the times to a 2D array for linear regression
t = t.reshape(-1, 1)

# Fit a linear regression model to the data
model = LinearRegression()
fit = model.fit(t,pos)

# Predict the positions at the given time stamps
new_timestamp = np.array([1])
new_timestamp = new_timestamp.reshape(-1, 1)
predicted_position = model.predict(new_timestamp)

# Extract the coefficients and intercept from the fitted model
ax_coefficient = model.coef_[0][0]
ay_coefficient = model.coef_[1][0]
az_coefficient = model.coef_[2][0]
intercept_x = model.intercept_[0]
intercept_y = model.intercept_[1]
intercept_z = model.intercept_[2]

# Give the function of the trajectory of the drone
print(f"x(t)={ax_coefficient:.2f}*t+{intercept_x:.2f}")
print(f"y(t)={ay_coefficient:.2f}*t+{intercept_y:.2f}")
print(f"z(t)={az_coefficient:.2f}*t+{intercept_z:.2f}")

# Plot the predicted trajectory of the drone
predicted_trajectory = model.predict(t)

# Create the 3D figure with the tracked positions and the predicted trajectory of the drone
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos[:,0],
           pos[:,1],
           pos[:,2],
           s=50,
           color='navy',
           label='Tracked Positions')
ax.plot(predicted_trajectory[:,0],
        predicted_trajectory[:,1],
        predicted_trajectory[:,2],
        linewidth=2,
        color='crimson',
        label='Predicted Trajectory')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Drone Trajectory',pad=20)
ax.legend()
plt.show()



# Linear regression formulas
# x(t)= -0.44*t + 1.47
# y(t)= -0.59*t + 2.13
# z(t)= 0.48*t + 0.80

