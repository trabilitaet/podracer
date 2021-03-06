import numpy as np
import numdifftools as nda
import math
import ipopt
from matplotlib import pyplot as plt

Nx = 7
r0 = np.array([0,0])
v0 = np.array([10,5])
phi0 = np.pi/2
r1 = np.array([250,500])


################################################################
# DEFINITIONS
# current state of VARIABLES x = [rx0,rx1,...vyNp-1,aNp-1,wNp-1]
# dim(x) = Np * 7
# to access variable i in timestep k: x[7*k+1]
# 0   1   2    3   4   5  6
# rx  ry  phi  vx  vy  a  w
################################################################

objective_all = lambda x : sum(pow((r1[0]-x[Nx*k+0]),2)+pow((r1[1]-x[Nx*k+1]),2) for k in range(Np-1,Np)) # only final
# objective_all = lambda x : sum(k**2*pow((r1[0]-x[Nx*k+0]),2)+k**2*pow((r1[1]-x[Nx*k+1]),2) for k in range(Np)) #entire time
first_order_constraint = lambda x,k,var1,var2 : x[Nx*(k+1)+var1] - x[Nx*k+var1] - x[Nx*k+var2]
constraint_vx = lambda x,k : x[Nx*(k+1)+3] - 0.85*x[Nx*k+3] - 0.85*x[Nx*k+5]*math.cos(x[Nx*k+2])
constraint_vy = lambda x,k : x[Nx*(k+1)+4] - 0.85*x[Nx*k+4] - 0.85*x[Nx*k+5]*math.sin(x[Nx*k+2])
constraint_ini = lambda x,j : x[j]

class nmpc_model():
    def __init__(self,r0,v0,r1,N_hat):
        self.Q = np.array([[1,0],[0,1]])

        self.x0 = r0
        self.v0 = v0
        self.r1 = r1
        self.N_hat = N_hat
        self.Np = N_hat +1
        self.n_constraints = 5*(Np-1)+5

        self.grad = nda.Gradient(objective_all)
        self.jac = nda.Jacobian(self.constraints)
    
    ##############################################################################################
    # game OBJECTIVE function value at x
    # RETURN a single VALUE
    ##############################################################################################
    def objective(self, x):
        return objective_all(x)


    ##############################################################################################
    # game dynamics expressed as CONSTRAINTS
    # RETURN a VECTOR of constraints values at x
    ##############################################################################################
    def constraints(self,x):
        constraints = np.array([])
        constraint = np.zeros((self.Np-1))
        for k in range(self.Np-1):
            constraint[k] = first_order_constraint(x,k,0,3)
        constraints = np.append(constraints,constraint)
        for k in range(self.Np-1):
            constraint[k] = first_order_constraint(x,k,1,4)
        constraints = np.append(constraints,constraint)
        for k in range(self.Np-1):
            constraint[k] = first_order_constraint(x,k,2,6)
        constraints = np.append(constraints,constraint)
        for k in range(self.Np-1):
            constraint[k] = constraint_vx(x,k)
        constraints = np.append(constraints,constraint)
        for k in range(self.Np-1):
            constraint[k] = constraint_vy(x,k)
        constraints = np.append(constraints,constraint)

        constraint = np.zeros((5))
        for j in range(5):
            constraint[j] = constraint_ini(x,j)
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

def bounds(r0,phi0,v0,Np):

    x_min = 00 #all checkpoints in [0,16000]
    x_max = 20000 #all checkpoints in [0,16000]
    y_min = 00 #all checkpoints in [0,9000]
    y_max = 10000 #all checkpoints in [0,9000]
    phi_lim = math.pi
    v_lim = 1000 #actual max velocity is 561 in x and y
    #a_min = -100 if test else 0
    a_min = 0
    a_max = 100
    w_lim = math.pi/10

    #upper and lower bounds on variables
    #rx,ry,phi,vx,vy,a,w
    lb = np.zeros((Np,7))
    ub = np.zeros((Np,7))
    x0 = np.zeros((Np,7))
    for k in range(Np):
        lb[k,:] = np.array([x_min, y_min, -phi_lim, -v_lim, -v_lim, a_min, -w_lim])
        ub[k,:] = np.array([x_max, y_max, phi_lim, v_lim, v_lim, a_max, w_lim])
        x0[k,:] = np.array([x_max, y_max, phi_lim, v_lim, v_lim, a_max, w_lim])
        x0[k,0] = (r1[0]-r0[0])/Np + r0[0]#rx
        x0[k,1] = (r1[1]-r0[1])/Np + r0[1]#ry
    lb = lb.reshape(7*Np,1)
    ub = ub.reshape(7*Np,1)
    x0 = ub.reshape(7*Np,1)

    #upper and lower bounds on constraint functions
    cl = np.zeros((n_constraints))
    cu = np.zeros((n_constraints))

    #values for cu, cl for x,y,vx,vy,psi constraints are already zero
    cl[5*(Np-1)] = cu[5*(Np-1)] =r0[0]
    cl[5*(Np-1)+1] = cu[5*(Np-1)+1] = r0[1]
    cl[5*(Np-1)+2] = cu[5*(Np-1)+2] = phi0
    cl[5*(Np-1)+3] = cu[5*(Np-1)+3] = v0[0]
    cl[5*(Np-1)+4] = cu[5*(Np-1)+4] = v0[1]

    return lb, ub, cl, cu, x0

def min_steps(x0,phi0,v0,r1):
    #minsteps
    v = v0
    x = x0

    t_stop = 0
    while np.linalg.norm(v) > 0:
        t_stop += 1
        x = x + v
        v[0] = int(0.85*v[0])
        v[1] = int(0.85*v[1])

    x1 = x
    #rotation time
    d0 = r1 - x0 # distance vector at start point
    dist0 = np.linalg.norm(d0)

    d1 = r1 - x1 # distance vector at stop point
    dist1 = np.linalg.norm(d1)

    phi0 = math.acos((d0[0]) / dist0) #angle of target at start
    phi1 = math.acos((d1[0]) / dist1) #angle of target at stop

    t_rot = math.ceil(10 * min(np.pi,abs(phi1-phi0)) / np.pi) #rotation time at max. +/- pi/10 per tick

    t_travel = 0
    while np.abs(x[0]-r1[0]) >= 100 and np.abs(x[1]-r1[1]) >= 100:
        print(x, v)
        t_travel += 1
        x = x + v
        v[0] = int(0.85*v[0] + 85*math.cos(phi1))
        v[1] = int(0.85*v[1] + 85*math.sin(phi1))

    print('x1: ', x1)
    print('r1: ', r1)
    print('dist: ', dist1)
    print('phi1: ', phi1)
    print('t_stop: ', t_stop)
    print('t_rot: ', t_rot)
    print('t_travel: ', t_travel)

    return math.ceil((max(t_stop, t_rot) + t_travel))


# N_hat = 5
N_hat = min_steps(r0,phi0,v0,r1)
print('N_hat: ', N_hat)

Np = N_hat + 1 #prediction horizon

n_constraints = 5*(Np-1)+5
model = nmpc_model(r0,v0,r1,N_hat)

lb, ub, cl, cu, x0  = bounds(r0,phi0,v0,Np)


nlp = ipopt.problem(
    n=7*Np,
    m=n_constraints,
    problem_obj=model,
    lb=lb,
    ub=ub,
    cl=cl,
    cu=cu
)

nlp.addOption('max_iter', 100)

#SOLVE nlp
sol, info = nlp.solve(x0)
np.save('solution', sol)

sol = sol.reshape(-1,7)
rx,ry,phi,vx,vy,a,w = sol[:,0],sol[:,1],sol[:,2],sol[:,3],sol[:,4],sol[:,5],sol[:,6]

#plot
plt.subplot(6,1,1)
plt.plot(rx,ry, 'ko-')
plt.plot(r0[0],r0[1], 'go')
plt.plot(r1[0],r1[1], 'bo')

index = 0
for x,y in zip(rx,ry):

    label = str(index)

    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
    index += 1


plt.subplot(6,1,2)
plt.plot(phi, 'ko-')
plt.xlabel("timesteps")
plt.ylabel("angle")

plt.subplot(6,1,3)
plt.plot(vx, 'ko-')
plt.ylabel("velocity in x")

plt.subplot(6,1,4)
plt.plot(vy, 'ko-')
plt.ylabel("velocity in y")

plt.subplot(6,1,5)
plt.plot(a, 'ko-')
plt.xlabel("timesteps")
plt.ylabel("acceleration")

plt.subplot(6,1,6)
plt.plot(w, 'ko-')
plt.xlabel("timesteps")
plt.ylabel("change in angle")

plt.show()