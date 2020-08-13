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
    def __init__(self,r0,v0,r1,Np):
        self.Q = np.array([[1,0],[0,1]])

        self.x0 = r0
        self.v0 = v0
        self.r1 = r1
        self.Np = Np
        self.n_constraints = 5*(Np-1)+5

        self.objective_all = lambda x : sum(pow((self.r1[0]-x[7*k+0]),2)+pow((self.r1[1]-x[7*k+1]),2) for k in range(self.Np-1,self.Np)) # only final
        # objective_all = lambda x : sum(k**2*pow((r1[0]-x[7*k+0]),2)+k**2*pow((r1[1]-x[7*k+1]),2) for k in range(Np)) #entire time
        

        self.constraint_rx = lambda x,k : x[7*(k+1)] - x[7*k] - x[7*k+3]
        self.constraint_ry = lambda x,k : x[7*(k+1)+1] - x[7*k+1] - x[7*k+4]
        self.constraint_phi = lambda x,k : x[7*(k+1)+2] - x[7*k+2] - x[7*k+6]
        self.constraint_vx = lambda x,k : x[7*(k+1)+3] - 0.85*x[7*k+3] - 0.85*x[7*k+5]*math.cos(x[7*k+2])
        self.constraint_vy = lambda x,k : x[7*(k+1)+4] - 0.85*x[7*k+4] - 0.85*x[7*k+5]*math.sin(x[7*k+2])
        self.constraint_ini = lambda x,j : x[j]

        self.grad = nda.Gradient(self.objective_all)
        self.jac = nda.Jacobian(self.constraints)

    
    ##############################################################################################
    # game OBJECTIVE function value at x
    # RETURN a single VALUE
    ##############################################################################################
    def objective(self, x):
        return self.objective_all(x)


    ##############################################################################################
    # game dynamics expressed as CONSTRAINTS
    # RETURN a VECTOR of constraints values at x
    ##############################################################################################
    def constraints(self,x):
        constraints = np.array([])
        constraint = np.zeros((self.Np-1))
        for k in range(self.Np-1):
            constraint[k] = self.constraint_rx(x,k)
        constraints = np.append(constraints,constraint)
        for k in range(self.Np-1):
            constraint[k] = self.constraint_ry(x,k)
        constraints = np.append(constraints,constraint)
        for k in range(self.Np-1):
            constraint[k] = self.constraint_phi(x,k)
        constraints = np.append(constraints,constraint)
        for k in range(self.Np-1):
            constraint[k] = self.constraint_vx(x,k)
        constraints = np.append(constraints,constraint)
        for k in range(self.Np-1):
            constraint[k] = self.constraint_vy(x,k)
        constraints = np.append(constraints,constraint)

        constraint = np.zeros((5))
        for j in range(5):
            constraint[j] = self.constraint_ini(x,j)
        constraints = np.append(constraints,constraint)
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