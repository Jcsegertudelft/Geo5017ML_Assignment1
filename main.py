'''
MAIN FILE

WORK ON COMPONENTS IN SEPERATE FILES
'''


import numpy as np
from Models import constv,consta

def sse(model,positions,time):
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

def main():
    T=np.array([1,2,3,4,5,6], dtype=float)
    P=np.array([
        [2.00,0.00,1.00],
        [1.08,1.68,2.38],
        [-0.83,1.82,2.49],
        [-1.97,0.28,2.15],
        [-1.31,-1.51,2.59],
        [0.57,-1.91,4.32]],
        dtype=float)

    mv=constv(v=[0.0,0.0,0.0],r0=[0.0,0.0,0.0])
    mv=gradient_descent(mv,P,T,learn_rate=1e-3)

    v_cap, r0_cap=mv.return_vars()
    print("v=",v_cap)
    print("SSE=", sse(mv,P,T))

    ma=consta(a=[0.0,0.0,0.0],r0=[0.0,0.0,0.0], v0=[0.0,0.0,0.0])
    ma=gradient_descent(ma,P,T,learn_rate=1e-4)

    a_cap,v0_cap,r0_cap=ma.return_vars()
    print("a=",a_cap)
    print("SSE=",sse(ma,P,T))

if __name__=='__main__':
    main()
