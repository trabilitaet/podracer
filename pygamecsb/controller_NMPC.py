import numpy as np
import math
from scipy.special import lambertw
import ipopt

class model():
    def __init__(self, N_hat, Np, x0, r1, r2):
        self.Q = np.array([[1,0],[0,1]])
        self.N_hat = N_hat
        self.Np = Np

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
            xk = np.array([x[3*k],x[3*k+1]])
            J += np.dot(np.dot(xk-self.r1, self.Q), np.transpose(xk-self.r1))
        for k in range(self.N_hat-1, self.Np):
            xk = np.array([x[3*k],x[3*k+1]])
            J += np.dot(np.dot(xk-self.r2, self.Q), np.transpose(xk-self.r2))
        return J

    def constraints(self,x): #TODO: handle edges
        # Callback function for evaluating constraint functions. 
        # The callback functions accepts one parameter: 
        #    x (value of the optimization variables at which the constraints are to be evaluated). 
        # The function should return the constraints values at the point x.
        #x represents all 2*Np variables
        # return an np array of constraints
        n_inputs = 2
        n_inits = 5
        n_states = 3
        constraints = np.zeros((1,n_inputs*self.Np+n_inits))
        
        #initial conditions
        conditions[0] = x[0] #rx0
        conditions[1] = x[1] #ry0
        conditions[3] = x[2] #psi0
        conditions[4] = x[2] - x[0] #v0x
        conditions[5] = x[3] - x[1] #v0y

        #set the constraints for states in t=[3,Np]
        for k in range(self.Np-3)
            # constraints on a -> affect r[t+2] and r[t+1]
            constraints[n_inits+n_inputs*k] = (17/20*x[n_states*k+6]-3/17*x[n_states*k+3]-x[n_states*k])/math.cos(x[n_states*k+2])
            # constraints on w 
            constraints[n_inits+n_inputs*k+1] = x[n_states*k+5] - x[n_states*k+2]
        #constraint on the last w
        constraints[n_inputs*(self.Np-2)] = x[n_states*(self.Np-2)+5] - x[3*(self.Np-2)+2]

        return constraints


    def gradient(self,x):
        #Callback function for evaluating gradient of objective function.
        #The callback functions accepts one parameter: 
        #   x (value of the optimization variables at which the gradient is to be evaluated). 
        #The function should return the gradient of the objective function at the point x.
        grad = np.zeros((3*Np,1))
        for k in range(Np):
            # x-coord
            grad[3*k] = 2*(x[3*k]-self.r1[0])
            # y-coord
            grad[3*k+1] = 2*(x[3*k+1]-self.r1[1])
            # angle
            grad[3*k+2] = 0

        return grad

    def jacobian(self,x):
        # Callback function for evaluating Jacobian of constraint functions.
        # The callback functions accepts one parameter:
        #    x (value of the optimization variables at which the jacobian is to be evaluated).
        # The function should return the values of the jacobian as calculated using x. 
        # The values should be returned as a 1-dim numpy array 
        #(using the same order as you used when specifying the sparsity structure)

        return jacobian


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