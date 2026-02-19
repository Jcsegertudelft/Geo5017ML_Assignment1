import numpy as np

class constv:
    def __init__(self,v,r0):
        self.v = np.array(v)
        self.r0 = np.array(r0)

    def value(self, t):

        return self.v*t + self.r0

    def grad_target_func(self, positions, t):
        grad_v = -2*np.sum([(position - self.value(time))*time for time,position in zip(t,positions)], axis =0)
        grad_r0 = -2*np.sum([position - self.value(time) for time,position in zip(t,positions)], axis =0)
        grad = np.append(grad_v,grad_r0)
        return grad

    def update(self,diff):
        self.v -= diff[:3]
        self.r0 -= diff[3:6]

    def return_vars(self):
        return [self.v,self.r0]


class consta:
    def __init__(self,a,v0,r0):
        self.a = np.array(a)
        self.v0 = np.array(v0)
        self.r0 = np.array(r0)

    def value(self, t):
        return 0.5*self.a*t**2 + self.v0*t + self.r0

    def grad_target_func(self, positions, t):
        grad_a = -np.sum([(position - self.value(time))*time**2 for time,position in zip(t,positions)], axis =0)
        grad_v0 = -2*np.sum([(position - self.value(time))*time for time,position in zip(t,positions)], axis =0)
        grad_r0 = -2*np.sum([position - self.value(time) for time,position in zip(t,positions)], axis =0)
        grad = np.append(grad_a,[grad_v0,grad_r0])
        return grad

    def update(self, diff):
        self.a -= diff[:3]
        self.v0 -= diff[3:6]
        self.r0 -= diff[6:9]

    def return_vars(self):
        return [self.a,self.v0,self.r0]

