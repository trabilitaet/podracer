import numpy as np
import math
import ipopt
from matplotlib import pyplot as plt
import numdifftools as nda

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
        self.r1 = np.zeros((2))
        self.Np = 8
        self.n_constraints = 5*(self.Np-1)+5

    def update_state(self, r1):
        self.r1 = r1
        

    ##############################################################################################
    # game OBJECTIVE function value at x
    # RETURN a single VALUE
    ##############################################################################################
    def objective(self, x):
        return sum(pow((self.r1[0]-x[7*k+0]),2)+pow((self.r1[1]-x[7*k+1]),2) for k in range(self.Np-1))


    ##############################################################################################
    # game dynamics expressed as CONSTRAINTS
    # RETURN a VECTOR of constraints values at x
    ##############################################################################################
    def constraints(self,x):
        constraints = np.zeros((self.n_constraints))
        constraint = np.zeros((self.Np-1))
        for k in range(self.Np-1):
            constraints[k] = x[7*(k+1)] - x[7*k] - x[7*k+3]
            constraints[(self.Np-1)+k] = x[7*(k+1)+1] - x[7*k+1] - x[7*k+4]
            constraints[2*(self.Np-1)+k] = x[7*(k+1)+2] - x[7*k+2] - x[7*k+6]
            constraints[3*(self.Np-1)+k] = x[7*(k+1)+3] - 0.85*x[7*k+3] - 0.85*x[7*k+5]*math.cos(x[7*k+2])
            constraints[4*(self.Np-1)+k] = x[7*(k+1)+4] - 0.85*x[7*k+4] - 0.85*x[7*k+5]*math.sin(x[7*k+2])
        
        for j in range(5):
            constraints[5*(self.Np-1)+j] = x[j]
        return constraints

    ##############################################################################################
    # GRADIENT of OBJECTIVE FUNCTION
    # RETURN a VECTOR of derivatives at x
    ##############################################################################################
    def gradient(self,x):
        grad = np.zeros((7*self.Np))
        for k in range(self.Np):
            grad[7*k+0] = 2*(x[7*k+0]-self.r1[0])
            grad[7*k+1] = 2*(x[7*k+1]-self.r1[1])
        return grad


    ##############################################################################################
    # JACOBIAN of CONSTRAINTS functions
    # build jacobian one line at a time
    # each line corresponds to one constraint (one timestep of a type of constraint)
    # each line contains derivatives for all Np*7 variables
    # RETURN a VECTOR of derivatives evaluated at x
    ##############################################################################################
    def jacobian(self,x):
        Np = self.Np
        n_constraints = 5*(Np-1)+5
        n_vars = 7*Np

        jacobian = np.zeros((n_constraints,n_vars)) #input,variable, init
        print(np.shape(jacobian))
        condition_index = 0

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
            tmp[7*k+2] =  0.85*x[7*k+5]*math.sin(x[7*k+2]) #psi[t]
            tmp[7*k+3] = -0.85 #vx[t]
            tmp[7*k+5] = -0.85*math.cos(x[7*k+2]) #a[t]
            tmp[7*k+10] = 1 #vx[t+1]
            jacobian[condition_index,:] = tmp
            condition_index += 1

        #TODO check signs for these
        for k in range(Np-1):
            tmp = np.zeros((n_vars))
            tmp[7*k+2] = -0.85*x[7*k+5]*math.cos(x[7*k+2]) #psi[t]
            tmp[7*k+4] = -0.85 #vy[t]
            tmp[7*k+5] = -0.85*math.sin(x[7*k+2]) #a[t]
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


