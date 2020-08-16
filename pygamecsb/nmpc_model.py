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
        self.Q = np.array([[1,0],[0,1]])

        self.grad = nda.Gradient(self.objective)
        self.jac = nda.Jacobian(self.constraints)
        # self.hess = nda.Hessian(self.compute_lagrangian)

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
        return self.grad(x)

    ##############################################################################################
    # JACOBIAN of CONSTRAINTS functions
    # build jacobian one line at a time
    # each line corresponds to one constraint (one timestep of a type of constraint)
    # each line contains derivatives for all Np*7 variables
    # RETURN a VECTOR of derivatives evaluated at x
    ##############################################################################################
    def jacobian(self,x):
        return self.jac(x)

    def compute_lagrangian(self,x):
        #lambdas attached to the end of x
        objective = pow((self.r1[0]-x[7*(self.Np-1)+0]),2)+pow((self.r1[1]-x[7*(self.Np-1)+1]),2)
        lagrangian = x[7*self.Np]*objective - np.dot(x[7*self.Np+1:], self.constraints(x))
        return lagrangian

    def hessian(self,x,lam,factor):
        x = np.append(x,factor)
        x = np.append(x,lam)
        return self.hess(x).reshape(-1,1)

