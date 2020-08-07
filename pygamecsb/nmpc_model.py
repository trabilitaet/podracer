import numpy as np
import math
import ipopt
from matplotlib import pyplot as plt

class nmpc_model():
    def __init__(self, Np):
        self.Q = np.array([[1,0],[0,1]])
        self.Np = Np
        self.N_hat = Np-1 #set to large value initially

        self.n_vars = 7 #rx,ry,psi,vx,vy,a,w
        self.n_constraints = 7*(self.Np-1)+5 #vars + inits

        self.r1 = np.array([0,0])
        self.r2 = np.array([0,0])

   # def __init__(self, N_hat, Np, x0, r1, r2):
   #     self.Q = np.array([[1,0],[0,1]])
   #     self.N_hat = N_hat
   #     self.Np = Np

   #     self.n_vars = 7 #rx,ry,psi,vx,vy,a,w
   #     self.n_constraints = 7*(self.Np-1)+5 #vars + inits

   #     self.r1 = r1
   #     self.r2 = r2

    def update(self, x0, v0, r1, r2, N_hat):
        # update values of x0, r1,r2, N_hat
        self.x0 = x0
        self.v0 = v0
        self.r1 = r1
        self.r2 = r2
        self.N_hat = N_hat

    # for state vector x, return value of objective function
    def objective(self, x):
        # Callback function for evaluating objective function. 
        # The callback functions accepts one parameter: 
        #     x (value of the optimization variables at which the objective is to be evaluated).
        # The function should return the objective function value at the point x.
        x = x.reshape(self.Np,7)
        J = 0
        for k in range(self.N_hat-1):
            rk = np.array([x[k,0],x[k,1]]) # extract x,y in this timestep
            J += np.dot(np.dot(rk-self.r1, self.Q), np.transpose(rk-self.r1)) # dist to next target
        for k in range(self.N_hat-1, self.Np):
            rk = np.array([x[k,0],x[k,1]]) # extract x,y in this timestep
            J += np.dot(np.dot(rk-self.r2, self.Q), np.transpose(rk-self.r2)) # dist to next target
        return J

    def constraints(self,x):
        # Callback function for evaluating constraint functions. 
        # The callback functions accepts one parameter: 
        #    x (value of the optimization variables at which the constraints are to be evaluated). 
        # The function should return the constraints values at the point x.
        x = x.reshape(-1,7)
        constraints = np.zeros((self.n_constraints))
        # for every type of constraint, add one for every timestep
        for k in range(self.Np-1):
            constraints[k] = x[k,5]                                                           #a
            constraints[k+1*self.Np-1] = x[k,6]                                               #w
            constraints[k+2*(self.Np-1)] = x[k+1,0] - x[k,0] - x[k,3]                         #x
            constraints[k+3*(self.Np-1)] = x[k+1,1] - x[k,1] - x[k,4]                         #y
            constraints[k+4*(self.Np-1)] = x[k+1,3]-0.85*x[k,3]-0.85*x[k,5]*math.cos(x[k,2])  #vx
            constraints[k+5*(self.Np-1)] = x[k+1,4]-0.85*x[k,4]-0.85*x[k,5]*math.sin(x[k,2])  #vy
            constraints[k+6*(self.Np-1)] = x[k+1,2]-x[k,2]-x[k,6] #psi
        
        for i in range(5):
            constraints[7*(self.Np-1)+i] = x[0,i] #initial conditions

        #goal reaching constraints
        #TODO

        return constraints


    def gradient(self,x):
        #Callback function for evaluating gradient of objective function.
        #The callback functions accepts one parameter: 
        #   x (value of the optimization variables at which the gradient is to be evaluated). 
        # The function should return the gradient of the objective function at the point x.
        x = x.reshape(self.Np,7)
        grad = np.zeros((self.Np,7))
        # only components for x,y are nonzero
        for k in range(self.Np):
            #x-coord
            grad[k,0] =  2*(x[k,0]-self.r1[0])
            #y-coord
            grad[k,1] =  2*(x[k,1]-self.r1[1])
        return grad.reshape(self.Np*7,1)

    def jacobian(self,x):
        # Callback function for evaluating Jacobian of constraint functions.
        # The callback functions accepts one parameter:
        #    x (value of the optimization variables at which the jacobian is to be evaluated).
        # The function should return the values of the jacobian as calculated using x. 
        # The values should be returned as a 1-dim numpy array 
        # (using the same order as you used when specifying the sparsity structure)
        # 7 vars -> 7Np derivatives for every constraint 
        x = x.reshape(self.Np,7)
        Np = self.Np
        n_constraints = 2*Np+5*(Np-1)+5
        n_vars = 7*Np
        jacobian = np.zeros((n_constraints,n_vars)) #input,variable, init
        condition_index = 0

        # Np constraints on Np timestep variables of a
        for k in range(Np):
            tmp = np.zeros((n_vars))
            tmp[7*k+5] = 1  
            jacobian[condition_index,:] = tmp
            condition_index +=1

        # Np constraints on Np variables of w
        for k in range(Np):
            tmp = np.zeros((n_vars))
            tmp[7*k+6] = 1  
            jacobian[condition_index,:] = tmp
            condition_index +=1

        # change in position constraint in x
        for k in range(Np-1):
            tmp = np.zeros((n_vars))
            tmp[7*k] = -1   #rx,t
            tmp[7*k+3] = -1 #vx,t
            tmp[7*k+7] = 1 #rx,t+1
            jacobian[condition_index,:] = tmp
            condition_index +=1

        # change in position constraint in y
        for k in range(Np-1):
            tmp = np.zeros((n_vars))
            tmp[7*k+1] = -1 #ry,t
            tmp[7*k+4] = -1 #vy,t
            tmp[7*k+8] = 1 #ry,t+1
            jacobian[condition_index,:] = tmp
            condition_index +=1

        # change in angle constraints psi
        for k in range(Np-1):
            tmp = np.zeros((n_vars))
            tmp[7*k+2] = -1 #phi,t
            tmp[7*k+6] = -1 #w,t
            tmp[7*k+9] = 1 #phi,t+1
            jacobian[condition_index,:] = tmp
            condition_index +=1

        # change in velocity constraints
        for k in range(Np-1):
            tmp = np.zeros((n_vars))
            tmp[7*k+2] =  0.85*x[k,5]*math.sin(x[k,2]) #psi[t]
            tmp[7*k+3] = -0.85 #vx[t]
            tmp[7*k+5] = -0.85*math.cos(x[k,2]) #a[t]
            tmp[7*k+10] = 1 #vx[t+1]
            jacobian[condition_index,:] = tmp
            condition_index += 1

        #TODO check signs for these
        for k in range(Np-1):
            tmp = np.zeros((n_vars))
            tmp[7*k+2] = 0.85*x[k,5]*math.cos(x[k,2]) #psi[t]
            tmp[7*k+4] = -0.85 #vy[t]
            tmp[7*k+5] = 0.85*math.sin(x[k,2]) #a[t]
            tmp[7*k+11] = 1 #vy[t+1]
            jacobian[condition_index,:] = tmp
            condition_index += 1

        # initial conditions
        for index in range(5):
            tmp = np.zeros(n_vars)
            tmp[index] = 1
            jacobian[condition_index,:] = tmp
            condition_index += 1
        return jacobian