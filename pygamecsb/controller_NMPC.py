import numpy as np
import math
from scipy.special import lambertw
import ipopt

class model():
    def __init__(self, Np, x, y, next_checkpoint_4x, next_checkpoint_y):
        self.Q = np.array([[1,0],[0,1]])
        self.Np = Np
        self.x = np.zeros((self.Np, 2))
        self.x[0,:] = x,y # ?

    # for state vector x, return value of objective function
    def objective(self, x):
        J = 0
        for 


    def constraints(self):
    def gradient(self):
    def jacobian(self):

class NMPC():
    def __init__(self,x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_angle):
        N_hat = self.min_steps(x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_angle)

        # set prediction horizon ?
        self.Np = N_hat + 1 #??7

        # checkpoints
        self.checkpoints = np.load('checkpoints.npy')
        self.n_checkpoints = self.checkpoints.shape[0]

        # lower and upper bounds for a,w
        self.lb = [0,-math.pi/10]
        self.ub = [math.pi/10, 100]

        self.n_var = 2
        # constraints on states for every time step
        self.n_constraints = 7*self.Np*2

        #set constraints on states, rx,ry,vx,vy,psi including initial condition

    def calculate(self, x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_angle):
        # TODO: make sure there are no rounding issues
        checkpointindex = self.get_checkpoint_index(next_checkpoint_x, next_checkpoint_y)
        r1 = self.checkpoints[checkpointindex,:]
        r2 = self.checkpoints[min(checkpointindex + 1, self.n_checkpoints),:]



        nlp = ipopt.problem(
            n=len(x1),
            m=len(cl),
            problem_obj=,
            lb=self.lb,
            ub=self.ub,
            cl=self.cl,
            cu=self.cu
        )






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