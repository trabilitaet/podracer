import numpy as np
import math
import ipopt
from matplotlib import pyplot as plt


r0 = np.array([1,1])
v0 = np.array([-1,-1])
phi0 = 2*np.pi/3
r1 = np.array([10,10])
r2 = np.array([0,20])

################################################################
# DEFINITIONS
# current state of VARIABLES x = [rx0,rx1,...vyNp-1,aNp-1,wNp-1]
# dim(x) = Np * 7
# to access variable i in timestep k: x[7*k+1]
# 0   1   2    3   4   5  6
# rx  ry  phi  vx  vy  a  w
################################################################

class nmpc_model():
    def __init__(self, n_constraints):
        self.Q = np.array([[1,0],[0,1]])
        self.Np = 0
        self.N_hat = 0 #set to large value initially

        self.n_constraints = n_constraints #5 state vars + inits

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
            rk = np.array([x[k,0],x[k,1]]) # extract rx,ry in this timestep
            J += np.dot(np.dot(rk-r1, self.Q), np.transpose(rk-r1)) 
        for k in range(self.N_hat-1, Np):
            rk = np.array([x[k,0],x[k,1]]) # extract rx,ry in this timestep
            J += np.dot(np.dot(rk-r2, self.Q), np.transpose(rk-r2))
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

#utility
def min_steps(x0, v0, deltaphi0, r1):
    x = x0
    v = v0

    t_stop = 0
    while np.linalg.norm(v) > 0:
        t_stop += 1
        x = x + v
        v[0] = int(0.85*v[0])
        v[1] = int(0.85*v[1])

    #x is now at stop position
    x1 = x
    print('x1: ', x1)
    print('r1: ', r1)
    #rotation time
    d0 = r1 - x0 # distance vector at start point
    dist0 = np.linalg.norm(d0)
    d1 = r1 - x1 # distance vector at stop point
    print('d1: ', d1)
    dist1 = np.linalg.norm(d1)
    print('dist: ', dist1)

    phi0 = math.acos((d0[0]) / dist0) #angle of target at start
    phi1 = math.acos((d1[0]) / dist1) #angle of target at stop
    print('phi1: ', phi1)

    deltaphi1 = (deltaphi0 + (phi0-phi1))%2*np.pi
    t_rot = math.ceil(10 * np.abs(deltaphi1) / math.pi) #rotation time at max. +/- pi/10 per tick
    print('t_rot: ', t_rot)

    t_travel = 0 #should change condition to sign change maybe? or nondecreasing?
    while np.abs(x[0]-r1[0]) <= 300 and np.abs(x[1]-r1[1]) <= 300:
        print(x, v)
        t_travel += 1
        x = x + v
        v[0] = int(0.85*v[0] + 85*math.cos(-phi1))
        v[1] = int(0.85*v[1] + 85*math.sin(-phi1))
    print('t_travel: ', t_travel)

    return max(t_stop, t_rot) + t_travel


N_hat = min_steps(r0, v0, phi0, r1)
print('N_hat: ', N_hat)

Np = N_hat + 1 #prediction horizon

n_constraints = 5*(Np-1)+5
model = nmpc_model(n_constraints)
model.update(r0, v0, r1, r2, N_hat)

x_min = -1000 #all checkpoints in [0,16000]
x_max = 20000 #all checkpoints in [0,16000]
y_min = -1000 #all checkpoints in [0,9000]
y_max = 10000 #all checkpoints in [0,9000]
phi_lim = math.pi
v_lim = 1000 #actual max velocity is 561 in x and y
#a_min = -100 if test else 0
a_min = 0
a_max = 10
w_lim = math.pi/10

#upper and lower bounds on variables
#rx,ry,phi,vx,vy,a,w
lb = np.zeros((Np,7))
ub = np.zeros((Np,7))
for k in range(Np):
    lb[k,:] = np.array([x_min, y_min, -phi_lim, -v_lim, -v_lim, a_min, -w_lim])
    ub[k,:] = np.array([x_max, y_max, phi_lim, v_lim, v_lim, a_max, w_lim])
lb = lb.reshape(7*Np,1)
ub = ub.reshape(7*Np,1)

#upper and lower bounds on constraint functions
cl = np.zeros((n_constraints))
cu = np.zeros((n_constraints))

#values for cu, cl for x,y,vx,vy,psi constraints are already zero
cl[5*(Np-1)] = r0[0]
cu[5*(Np-1)] = r0[0]
cl[5*(Np-1)+1] = r0[1]
cu[5*(Np-1)+1] = r0[1]
cl[5*(Np-1)+2] = phi0
cu[5*(Np-1)+2] = phi0
cl[5*(Np-1)+3] = v0[0]
cu[5*(Np-1)+3] = v0[0]
cl[5*(Np-1)+4] = v0[1]
cu[5*(Np-1)+4] = v0[1]
#values for v0 constraints are already zero

x0 = ub/2

nlp = ipopt.problem(
    n=7*Np,
    m=n_constraints,
    problem_obj=model,
    lb=lb,
    ub=ub,
    cl=cl,
    cu=cu
)

# nlp.addOption('max_iter', 6000)

#SOLVE nlp
x, info = nlp.solve(x0)
print(info)
# plt.imshow(x.reshape(Np,7))
# plt.show()
x = x.reshape(-1,7)
rx = x[:,0]
print('rx: ', rx)
ry = x[:,1]
print('ry: ', ry)
phi = x[:,2]
print('phi: ', phi)
vx = x[:,3]
print('vx: ', vx)
vy = x[:,4]
print('vy: ', vy)
a = x[:,5]
print('a: ', a)
w = x[:,6]
print('w: ', w)

plt.plot(rx,ry, 'ro')
plt.show()