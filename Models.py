import numpy as np

class constv: #Model constant velocity
    def __init__(self,v,r0):
        self.v = np.array(v)
        self.r0 = np.array(r0)

    def value(self, t):
        #Return the value at time = t
        return self.v*t + self.r0

    def grad_target_func(self, positions, t):
        #Return the gradient of the error function with respect to the unknowns
        #Given a set of positions and times
        grad_v = -2*np.sum([(position - self.value(time))*time for time,position in zip(t,positions)], axis =0)
        grad_r0 = -2*np.sum([position - self.value(time) for time,position in zip(t,positions)], axis =0)
        grad = np.append(grad_v,grad_r0)
        return grad

    def update(self,diff):
        #update the value of the model from the difference vector
        self.v -= diff[:3]
        self.r0 -= diff[3:6]

    def return_vars(self):
        #Return all the unknowns at their current value
        return [self.v,self.r0]


class consta: #Model constant acceleration
    def __init__(self,a,v0,r0):
        self.a = np.array(a)
        self.v0 = np.array(v0)
        self.r0 = np.array(r0)

    def value(self, t):
        #Return the value at time = t
        return 0.5*self.a*t**2 + self.v0*t + self.r0

    def grad_target_func(self, positions, t):
        #Return the gradient of the error function with respect to the unknowns
        #Given a set of positions and times
        grad_a = -np.sum([(position - self.value(time))*time**2 for time,position in zip(t,positions)], axis =0)
        grad_v0 = -2*np.sum([(position - self.value(time))*time for time,position in zip(t,positions)], axis =0)
        grad_r0 = -2*np.sum([position - self.value(time) for time,position in zip(t,positions)], axis =0)
        grad = np.append(grad_a,[grad_v0,grad_r0])
        return grad

    def update(self, diff):
        #update the value of the model from the difference vector
        self.a -= diff[:3]
        self.v0 -= diff[3:6]
        self.r0 -= diff[6:9]

    def return_vars(self):
        #Return all the unknowns at their current value
        return [self.a,self.v0,self.r0]

