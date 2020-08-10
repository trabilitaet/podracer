import numpy as np
import math
import ipopt
from matplotlib import pyplot as plt

################################################################
# DEFINITIONS
# current state of VARIABLES x = [rx0,rx1,...vyNp-1,aNp-1,wNp-1]
# dim(x) = Np * 7
# to access variable i in timestep k: x[7*k+1]
# 0   1   2    3   4   5  6
# rx  ry  phi  vx  vy  a  w
################################################################

class nmpc_model():
    def __init__(self):
        self.Q = np.array([[1,0],[0,1]])
        self.Np = 0
        self.N_hat = 0 #set to large value initially

        self.r1 = np.array([0,0])
        self.r2 = np.array([0,0])


    ##############################################################################################
    # feed current game state to system model
    # SET internal values
    ##############################################################################################
    def update(self, r0, v0, r1, r2, N_hat):
        # update values of r0, r1,r2, N_hat
        self.x0 = r0
        self.v0 = v0
        self.r1 = r1
        self.r2 = r2
        self.N_hat = N_hat
        self.Np = N_hat +1

        self.n_constraints = 5*(self.Np-1)+5

    
    ##############################################################################################
    # game OBJECTIVE function value at x
    # RETURN a single VALUE
    ##############################################################################################
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


    ##############################################################################################
    # game dynamics expressed as CONSTRAINTS
    # RETURN a VECTOR of constraints values at x
    ##############################################################################################
    def constraints(self,x):
        # Callback function for evaluating constraint functions. 
        # The callback functions accepts one parameter: 
        #    x (value of the optimization variables at which the constraints are to be evaluated). 
        # The function should return the constraints values at the point x.
        x = x.reshape(-1,7)
        Np = self.Np
        constraints = np.zeros((self.n_constraints))
        # for every type of constraint, add one for every timestep
        for k in range(Np-1):
            constraints[k] = x[k+1,0] - x[k,0] - x[k,3]                                  #x   (I)
            constraints[k+(Np-1)] = x[k+1,1] - x[k,1] - x[k,4]                           #y   (II)
            constraints[k+2*(Np-1)] = x[k+1,2]-x[k,2]-x[k,6]                             #psi (III)
            constraints[k+3*(Np-1)] = x[k+1,3]-0.85*x[k,3]-0.85*x[k,5]*math.cos(x[k,2])  #vx  (IV) -> consider adding rounding
            constraints[k+4*(Np-1)] = x[k+1,4]-0.85*x[k,4]-0.85*x[k,5]*math.sin(x[k,2])  #vy  (V)
        
        for i in range(5):
            constraints[5*(Np-1)+i] = x[0,i] #initial conditions (VI)

        #goal reaching constraints
        #TODO

        return constraints

    ##############################################################################################
    # GRADIENT of OBJECTIVE FUNCTION
    # RETURN a VECTOR of derivatives at x
    ##############################################################################################
    def gradient(self,x):
        #Callback function for evaluating gradient of objective function.
        #The callback functions accepts one parameter: 
        #   x (value of the optimization variables at which the gradient is to be evaluated). 
        # The function should return the gradient of the objective function at the point x.
        Np = self.Np
        grad = np.zeros((Np*7))
        # only components for rx,ry are nonzero
        for k in range(self.N_hat):
            grad[k*7+0] =  2*(x[7*k+0]-self.r1[0]) #rx
            grad[k*7+1] =  2*(x[7*k+1]-self.r1[1]) #ry
        for k in range(self.N_hat, Np):
        #    grad[k*7+0] =  2*(x[7*k+0]-self.r2[0]) #rx
        #    grad[k*7+1] =  2*(x[7*k+1]-self.r2[1]) #ry
            grad[k*7+0] =  2*(x[7*k+0]-self.r1[0]) #rx
            grad[k*7+1] =  2*(x[7*k+1]-self.r1[1]) #ry
        return grad

    ##############################################################################################
    # JACOBIAN of CONSTRAINTS functions
    # build jacobian one line at a time
    # each line corresponds to one constraint (one timestep of a type of constraint)
    # each line contains derivatives for all Np*7 variables
    # RETURN a VECTOR of derivatives evaluated at x
    ##############################################################################################
    def jacobian(self,x):
        # Callback function for evaluating Jacobian of constraint functions.
        # The callback functions accepts one parameter:
        #    x (value of the optimization variables at which the jacobian is to be evaluated).
        # The function should return the values of the jacobian as calculated using x. 
        # The values should be returned as a 1-dim numpy array 
        # (using the same order as you used when specifying the sparsity structure)
        # 7 vars -> 7Np derivatives for every constraint 
        Np = self.Np
        x = x.reshape(Np,7)
        n_constraints = self.n_constraints #variable, init
        n_vars = 7*Np
        jacobian = np.zeros((n_constraints,n_vars))
        condition_index = 0

        # change in position constraint in x (I)
        for k in range(Np-1):
            tmp = np.zeros((n_vars))
            tmp[7*k] = -1   #rx,t
            tmp[7*k+3] = -1 #vx,t
            tmp[7*(k+1)] = 1 #rx,t+1
            jacobian[condition_index,:] = tmp
            condition_index +=1

        # change in position constraint in y (II)
        for k in range(Np-1):
            tmp = np.zeros((n_vars))
            tmp[7*k+1] = -1 #ry,t
            tmp[7*k+4] = -1 #vy,t
            tmp[7*(k+1)+1] = 1 #ry,t+1
            jacobian[condition_index,:] = tmp
            condition_index +=1

        # change in angle constraints psi (III)
        for k in range(Np-1):
            tmp = np.zeros((n_vars))
            tmp[7*k+2] = -1 #phi,t
            tmp[7*k+6] = -1 #w,t
            tmp[7*(k+1)+2] = 1 #phi,t+1
            jacobian[condition_index,:] = tmp
            condition_index +=1

        # change in velocity constraints (IV)
        for k in range(Np-1):
            tmp = np.zeros((n_vars))
            tmp[7*k+2] =  0.85*x[k,5]*math.sin(x[k,2]) #psi[t]
            tmp[7*k+3] = -0.85 #vx[t]
            tmp[7*k+5] = -0.85*math.cos(x[k,2]) #a[t]
            tmp[7*(k+1)+3] = 1 #vx[t+1]
            jacobian[condition_index,:] = tmp
            condition_index += 1

        #TODO check signs for these (V)
        for k in range(Np-1):
            tmp = np.zeros((n_vars))
            tmp[7*k+2] = 0.85*x[k,5]*math.cos(x[k,2]) #psi[t]
            tmp[7*k+4] = -0.85 #vy[t]
            tmp[7*k+5] = 0.85*math.sin(x[k,2]) #a[t]
            tmp[7*(k+1)+4] = 1 #vy[t+1]
            jacobian[condition_index,:] = tmp
            condition_index += 1

        # initial conditions (VI)
        for index in range(5):
            tmp = np.zeros(n_vars)
            tmp[index] = 1
            jacobian[condition_index,:] = tmp
            condition_index += 1

        return jacobian