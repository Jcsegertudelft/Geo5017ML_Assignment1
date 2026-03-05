'''
Code Group 10: Job Segers, Timber Groeneveld, Akhil Veeranki
'''

import numpy as np
from  matplotlib import pyplot as plt
from Models import constv,consta

Model_used = 'consta' #set to constv for constant velocity, to consta for constant acceleration
Learning_rate = 0.001
Max_iter = 100000
Tolerance = 1e-6 #Minimum step size for another iteration to take place
Viz = True #Do you want the plot at the end. True or False

def sse(model,positions,time): #sum of squared errors
    pred=np.array([model.value(t) for t in time])
    return np.sum((positions - pred)**2)


def gradient_descent(model,positions,time,learn_rate=1e-3,max_iter=100000,tol=1e-6):
    for it in range(max_iter):
        grad=model.grad_target_func(positions,time)
        diff=learn_rate * grad
        if np.linalg.norm(diff) < tol:
            break
        model.update(diff)

    return model


def create_model(model_type):
    if model_type == "constv":
        mv=constv(v=[0.0,0.0,0.0],r0=[0.0,0.0,0.0])
        return mv

    elif model_type == "consta":
        ma=consta(a=[0.0,0.0,0.0],r0=[0.0,0.0,0.0], v0=[0.0,0.0,0.0])
        return ma
    else:
        raise Exception("Model type not recognized, input 'constv' or 'consta'")
        return None


def main(Viz):
    # Define the time points and positions
    T=np.array([1,2,3,4,5,6], dtype=float)
    P=np.array([
        [2.00,0.00,1.00],
        [1.08,1.68,2.38],
        [-0.83,1.82,2.49],
        [-1.97,0.28,2.15],
        [-1.31,-1.51,2.59],
        [0.57,-1.91,4.32]],
        dtype=float)

    # Create the model with either constv or consta
    model = create_model(Model_used)
    # Train the model using gradient descent
    lr_gradient_descent = gradient_descent(model,P,T,learn_rate=Learning_rate,max_iter=Max_iter,tol=Tolerance)
    # Print the parameters
    gd_output = lr_gradient_descent.return_vars()
    rounded_gd_output = [np.round(arr, 4) for arr in gd_output]
    print("Linear Regression (Gradient_descent): ", rounded_gd_output)
    print("SSE=", sse(model, P, T))
    # Predict the trajectory of the drone using the parameters obtained from gradient descent with constant velocity including at t=7
    predicted_positions_gd = [lr_gradient_descent.value(time) for time in T]
    predicted_positions_gd.append(lr_gradient_descent.value(7))
    predicted_positions_gd = np.array(predicted_positions_gd)
    print("Predicted positions including at t=7:", predicted_positions_gd)

    # Extract the x, y and z coordinates of the predicted positions for plotting
    x_predicted = predicted_positions_gd[:,0]
    y_predicted = predicted_positions_gd[:,1]
    z_predicted = predicted_positions_gd[:,2]

    # Extract the x, y, z coordinates and time stamps
    x_measured = P[:,0]
    y_measured = P[:,1]
    z_measured = P[:,2]

    # Create the 3D figure with the tracked positions and the trajectory from gradient descent
    if Viz:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_measured,y_measured,z_measured, color='navy', label='Measured Positions')
        ax.scatter(x_predicted[-1],y_predicted[-1],z_predicted[-1], color='navy', label='Predicted Position at t=7', s=80)
        ax.plot(x_predicted,y_predicted,z_predicted, linewidth=1.4, color='crimson', label='Fitted Trajectory')
        ax.set_xlabel('X Position', fontsize=13)
        ax.set_ylabel('Y Position', fontsize=13)
        ax.set_zlabel('Z Position', fontsize=13)
        ax.set_title('Fitted Drone Trajectory using Gradient Descent', pad=20)
        ax.text(x_measured[0]+ 0.2,y_measured[0],z_measured[0],'t = 1', fontsize=12)
        ax.text(x_measured[1]+ 0.2,y_measured[1],z_measured[1],'t = 2', fontsize=12)
        ax.text(x_measured[2]+ 0.2,y_measured[2],z_measured[2],'t = 3', fontsize=12)
        ax.text(x_measured[3]+ 0.2,y_measured[3],z_measured[3],'t = 4', fontsize=12)
        ax.text(x_measured[4]+ 0.2,y_measured[4],z_measured[4],'t = 5', fontsize=12)
        ax.text(x_measured[5]+ 0.2,y_measured[5],z_measured[5],'t = 6', fontsize=12)
        ax.text(x_predicted[-1]+ 0.35,y_predicted[-1],z_predicted[-1],'t = 7',fontsize=13)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-5, 2)
        ax.set_aspect('equal')
        ax.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('Drone_Trajectory_With_Fit.png', dpi = 300)
        plt.show()


if __name__=='__main__':
    main(Viz)
