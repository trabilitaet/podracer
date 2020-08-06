import numpy as np
import math
from matplotlib import pyplot as plt

Np = 5
x = 2*np.ones((Np,7))
jacobian = np.zeros((2*7*Np+4*7*Np*(Np-1)+5*7*Np,1)) #input,variable, init
print(np.shape(jacobian))
offset = 0
# input constraints: one matrix per time step
tmp = np.zeros((Np,7))
for k in range(Np):
    tmp[k,5] = 1 #a
jacobian[0:Np*7] = tmp.flatten().reshape((7*Np,1))
offset += Np*7
print(offset)

tmp = np.zeros((Np,7))
for k in range(Np):
    tmp[k,6] = 1 #w
jacobian[offset:offset+Np*7] = tmp.flatten().reshape((7*Np,1))
offset += Np*7
print(offset)

#velocity constraints: one matrix per time step per constraint
for k in range(Np-1):
    tmp = np.zeros((Np,7))
    tmp[k,0] = -1 #-rx[t]
    tmp[k,2] = -1 #-vx[t]
    tmp[k+1,0] =  1 #rx[t+1]
    jacobian[offset:offset+Np*7] = tmp.flatten().reshape((7*Np,1))
    offset += Np*7
print(offset)

for k in range(Np-1):
    tmp = np.zeros((Np,7))
    tmp[k,1] = -1 #-ry[t]
    tmp[k,3] = -1 #-vy[t]
    tmp[k+1,1] =  1 #ry[t+1]
    jacobian[offset:offset+Np*7] = tmp.flatten().reshape((7*Np,1))
    offset += Np*7
print(offset)

#accelleration constraints
for k in range(Np-1):
    tmp = np.zeros((Np,7))
    tmp[k,2] = 0.85*x[k,5]*math.sin(x[k,2]) #psi[t]
    tmp[k,3] = -0.85 #vx[t]
    tmp[k,5] = -0.85*math.cos(x[k,2]) #a[t]
    tmp[k+1,3] = 1 #vx[t+1]
    jacobian[offset:offset+Np*7] = tmp.flatten().reshape((7*Np,1))
    offset += Np*7
print(offset)

for k in range(Np-1):#check equations
    tmp = np.zeros((Np,7))
    tmp[k,2] = -0.85*x[k,5]*math.cos(x[k,2]) #psi[t]
    tmp[k,4] = -0.85 #vy[t]
    tmp[k,5] = 0.85*math.sin(x[k,2]) #a[t]
    tmp[k+1,4] = 1 #vy[t+1]
    jacobian[offset:offset+Np*7] = tmp.flatten().reshape((7*Np,1))
    offset += Np*7
print(offset)

#initial conditions
# 5 constraints -> 5*7*Np derivatives
for i in range(5):
    tmp = np.zeros((Np,7))
    tmp[0,i] = 1
    jacobian[offset:offset+Np*7] = tmp.flatten().reshape((7*Np,1))
    offset += Np*7
print(offset)

heatmap = jacobian.reshape((int(np.shape(jacobian)[0]/7),7))
plt.imshow(heatmap)
plt.show()