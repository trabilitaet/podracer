import numpy as np
import math
from scipy.special import lambertw
import ipopt

class model():
    def __init__(self, N_hat, Np, x0, r1, r2):
        self.Q = np.array([[1,0],[0,1]])
        self.N_hat = N_hat
        self.Np = Np

        self.n_vars = 7 #rx,ry,psi,vx,vy,a,w
        self.n_constraints = 7*(self.Np-1)+5 #vars + inits

        self.r1 = r1
        self.r2 = r2

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
        J = 0
        for k in range(self.N_hat-1):
            rk = np.array([x[k,0],x[k,1]]) # extract x,y in this timestep
            J += np.dot(np.dot(xk-self.r1, self.Q), np.transpose(xk-self.r1)) # dist to next target
        for k in range(self.N_hat-1, self.Np):
            rk = np.array([x[k,0],x[k,1]]) # extract x,y in this timestep
            J += np.dot(np.dot(xk-self.r2, self.Q), np.transpose(xk-self.r2)) # dist to next target
        return J

    def constraints(self,x):
        # Callback function for evaluating constraint functions. 
        # The callback functions accepts one parameter: 
        #    x (value of the optimization variables at which the constraints are to be evaluated). 
        # The function should return the constraints values at the point x.
        constraints = np.zeros((self.n_constraints))
        # for every type of constraint, add one for every timestep
        for k in range(self.Np-1):
            #input constraints
            constraints[k] = x[k,5] #a
            constraints[k+1*self.Np-1] = x[k,6] #w
            #velocity constraints
            constraints[k+2*(self.Np-1)] = x[k+1,0] - x[k,0] - x[k,3] #x
            constraints[k+3*(self.Np-1)] = x[k+1,1] - x[k,1] - x[k,4] #y
            #accelleration constraints
            constraints[k+4*(self.Np-1)] = x[k+1,3]-0.85*x[k,3]-0.85*x[k,5]*math.cos(x[k,2]) #vx
            constraints[k+5*(self.Np-1)] = x[k+1,4]-0.85*x[k,4]-0.85*x[k,5]*math.sin(x[k,2]) #vy
            #angle constraints
            constraints[k+6*(self.Np-1)] = x[k+1,2]-x[k,2]-x[k,6] #psi
        #initial conditions
        
        #goal reaching constraints
        #TODO

        return constraints


    def gradient(self,x):
        #Callback function for evaluating gradient of objective function.
        #The callback functions accepts one parameter: 
        #   x (value of the optimization variables at which the gradient is to be evaluated). 
        # The function should return the gradient of the objective function at the point x.
        grad = np.zeros((self.Np,self.n_var))
        # only components for x,y are nonzero
        for k in range(Np):
            #x-coord
            grad[k,0] =  2*(x[k,0]-self.r1[0])
            #y-coord
            grad[k,1] =  2*(x[k,1]-self.r1[1])
        return grad

    def jacobian(self,x):
        # Callback function for evaluating Jacobian of constraint functions.
        # The callback functions accepts one parameter:
        #    x (value of the optimization variables at which the jacobian is to be evaluated).
        # The function should return the values of the jacobian as calculated using x. 
        # The values should be returned as a 1-dim numpy array 
        # (using the same order as you used when specifying the sparsity structure)
        # 7 vars -> 7Np derivatives for every constraint 
        Np = self.Np
        n_constraints = self.n_constraints
        jacobian = np.zeros((2*7*Np+4*7*(Np-1)+5*7*Np,1)) #input,variable, init
        offset = 0
        # input constraints: one matrix per time step
        tmp = np.zeros((Np,7))
        for k in range(Np):
            tmp[k,5] = 1 #a
        jacobian[0:Np*7] = tmp.flatten().reshape((7*Np,1))
        offset += Np*7

        tmp = np.zeros((Np,7))
        for k in range(Np):
            tmp[k,6] = 1 #w
        jacobian[offset:offset+Np*7] = tmp.flatten().reshape((7*Np,1))
        offset += Np*7

        #velocity constraints: one matrix per time step per constraint
        for k in range(Np-1):
            tmp = np.zeros((Np,7))
            tmp[k,0] = -1 #-rx[t]
            tmp[k,2] = -1 #-vx[t]
            tmp[k+1,0] =  1 #rx[t+1]
            jacobian[offset:offset+Np*7] = tmp.flatten().reshape((7*Np,1))
            offset += Np*7

        for k in range(Np-1):
            tmp = np.zeros((Np,7))
            tmp[k,1] = -1 #-ry[t]
            tmp[k,3] = -1 #-vy[t]
            tmp[k+1,1] =  1 #ry[t+1]
            jacobian[offset:offset+Np*7] = tmp.flatten().reshape((7*Np,1))
            offset += Np*7

        #accelleration constraints
        for k in range(Np-1):
            tmp = np.zeros((Np,7))
            tmp[k,2] = 0.85*x[k,5]*math.sin(x[k,2]) #psi[t]
            tmp[k,3] = -0.85 #vx[t]
            tmp[k,5] = -0.85*math.cos(x[k,2]) #a[t]
            tmp[k+1,3] = 1 #vx[t+1]
            jacobian[offset:offset+Np*7] = tmp.flatten().reshape((7*Np,1))
            offset += Np*7

        for k in range(Np-1):#check equations
            tmp = np.zeros((Np,7))
            tmp[k,2] = -0.85*x[k,5]*math.cos(x[k,2]) #psi[t]
            tmp[k,4] = -0.85 #vy[t]
            tmp[k,5] = 0.85*math.sin(x[k,2]) #a[t]
            tmp[k+1,4] = 1 #vy[t+1]
            jacobian[offset:offset+Np*7] = tmp.flatten().reshape((7*Np,1))
            offset += Np*7

        #initial conditions
        # 5 constraints -> 5*7*Np derivatives
        for i in range(5):
            tmp = np.zeros((Np,7))
            tmp[0,i] = 1
            jacobian[offset:offset+Np*7] = tmp.flatten().reshape((7*Np,1))
            offset += Np*7

        heatmap = jacobian.reshape((int(np.shape(jacobian)[0]/7),7))
        plt.imshow(heatmap)
        plt.show()


        return J


    def intermediate(self,x):
        # Optional. 
        # Callback function that is called once per iteration (during the convergence check),
        # and can be used to obtain information about the optimization status while IPOPT solves the problem.
        # If this callback returns False, IPOPT will terminate with the User_Requested_Stop status.
        # The information below corresponeds to the argument list passed to this callback:

class NMPC():
    def __init__(self):
        # checkpoints
        self.checkpoints = np.load('checkpoints.npy')
        self.n_checkpoints = self.checkpoints.shape[0]

        self.N_hat = self.min_steps(self.checkpoints[0,0], self.checkpoints[0,1], self.checkpoints[1,0], self.checkpoints[1,1])

        # set prediction horizon ?
        self.Np = self.N_hat + 1 #??

        # lower and upper bounds for a,w
        self.lb = [0,-math.pi/10]
        self.ub = [math.pi/10, 100]

        self.n_var = 2
        # constraints on states for every time step
        self.n_constraints = 7*self.Np*2
        self.model = model(self.N_hat, self.Np)
        self.old_checkpoint = np.zeros((2,1))

    # calculate is called in every step
    def calculate(self, rx, ry, next_checkpoint_x, next_checkpoint_y, next_checkpoint_angle):
        x = np.array([rx,ry])
        # TODO: make sure there are no rounding issues
        checkpointindex = self.get_checkpoint_index(next_checkpoint_x, next_checkpoint_y)
        r1 = self.checkpoints[checkpointindex,:]
        r2 = self.checkpoints[min(checkpointindex + 1, self.n_checkpoints),:]

        self.N_hat = max(1, self.N_hat -1)
        self.model.update(x, r1, r2, self.N_hat)

        nlp = ipopt.problem(
            n=len(x1),
            m=len(cl),
            problem_obj=self.model,
            lb=self.lb,
            ub=self.ub,
            cl=self.cl,
            cu=self.cu
        )


        # how to detect next checkpoint

        self.old_checkpoint = r1

        return thrust, next_checkpoint_x, next_checkpoint_y


    def get_checkpoint_index(self, checkpoint_x, checkpoint_y):
        for index in range(self.n_checkpoints):
            if self.checkpoints[index,0] == checkpoint_x and self.checkpoints[index,1] == checkpoint_y:
                return index
        return -1

    def min_steps(self,x0, v0, phi0, r1):

        #velocity decay time
        t1 = math.ceil(math.log(np.linalg.norm(v0)) / math.log(20 / 17))

        # position as velocity reaches 0
        x1 = x0
        for i in range(t1):
            x1 = x1 + pow(17/20,i)*v0

        # distance vector
        d = r1 - x1
        dist = np.linalg.norm(d)

        # calculate rotation time
        # move at max. +/- Pi/10 per tick
        # alpha: angle to target after velocity decay
        angle = math.acos((d[0]) / dist)
        print('angle: ', angle)
        t2 = math.ceil(10 * angle / math.pi)
        print('t2: ', t2)

        # calculate travel time from x1 to r1 at max acelleration (formula from mathematica)
        t3 = 17 / 3
        t3 += 3 * dist / 17
        t3 += lambertw(-1/3*pow(2,-34/3-6*dist/17) \
                       *pow(5,-17/3-3*dist/17)*pow(17,20/3+3*dist/17)*math.log(20/17))/math.log(20/17)
        t3 = math.ceil(t3)
        print('t3: ', t3)

        return max(t1,t2) + t3

    def getName(self):
        return 'NMPC'
