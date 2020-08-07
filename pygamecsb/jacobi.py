import numpy as np
import math
from matplotlib import pyplot as plt

Np = 5
x = 2*np.ones((Np,7))
n_constraints = 2*Np+5*(Np-1)+5
n_vars = 7*Np
jacobian = np.zeros((n_constraints,n_vars)) #input,variable, init
print(np.shape(jacobian))
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
    tmp[7*k+2] = -0.85*x[k,5]*math.cos(x[k,2]) #psi[t]
    tmp[7*k+4] = -0.85 #vy[t]
    tmp[7*k+5] = -0.85*math.sin(x[k,2]) #a[t]
    tmp[7*k+11] = 1 #vy[t+1]
    jacobian[condition_index,:] = tmp
    condition_index += 1

# initial conditions
for index in range(5):
    tmp = np.zeros(n_vars)
    tmp[index] = 1
    jacobian[condition_index,:] = tmp
    condition_index += 1

plt.imshow(jacobian)
plt.show()
