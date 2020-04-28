import numpy as np
import pickle as pkl

def f(A,b,x):
    return pow(np.linalg.norm(A @ x - b),2)

def f_p(A,b,x):
     return 2 * A.T @ (A @ x - b)

def g(x,lamb):
    return lamb * np.sum(np.abs(x))

def sthr(x,lamb):
    return np.sign(x) * (np.abs(x) - lamb).clip(0)

def p(A,b,x,lamb,t):
     return sthr(x - t * f_p(A,b,x),lamb)

def v(t,x_k_1,x_k_2,k):
    return x_k_1 + (k-2)/(k+1) * (x_k_1 - x_k_2)

def F(A,b,x,lamb):
    return f(A,b,x) + g(x,lamb)

def G(A,b,x,lamb,t):
    return (x - p(A,b,x,lamb,t))/t

def Q(A,b,x,fpy,lamb,t):
    Gx = G(A,b,x,lamb,t)
    return f(A,b,x) - t * fpy @ Gx + t/2 * Gx @ Gx

#def Q(A,b,x,y,fy,fpy,lamb,t):
#    return fy - (x - y) @ fpy + 1/(t*2) * (x - y).T @ (x-y) + g(x,lamb)

def backtracking(A,b,x,lamb,t0=1,eta=0.8):
    fpx = f_p(A,b,x) 
    t = t0
    while f(A,b,x-t*G(A,b,x,lamb,t))> Q(A,b,x,fpx,lamb,t):
        t = t * eta
    return t

def training(A,b,xinit,lamb,maxiter=1000):
    x = (xinit,None)
    t = (1,None)
    y = x[0]
    loss = []
    for i in range(maxiter):
        print(i)
        L = backtracking(A,b,y,lamb)
        x = (p(A,b,y,lamb,L),x[0])
        t = ((1 + np.sqrt(1 + 4 * pow(t[0],2)))/2,t[0])
        #y = v(t,x[0],x[1],i)
        y = x[0] + (t[1]-1)/t[0]*(x[0] - x[1])
        loss.append(F(A,b,x[0],lamb))
        print("Loss: {}, t={}, L={}".format(loss[-1],(t[1]-1)/t[0],L))
    return x[0],loss

