import numpy as np
import math
import ipopt
import nmpc_model
from matplotlib import pyplot as plt


class NMPC():
    ########################################################################
    # INIT for NMPC controller
    # called once before the start of the game
    # given current game state, output desired thrust and heading
    ########################################################################
    def __init__(self, test, x0, y0, delta_angle_0):
        self.old_x = x0
        self.old_y = y0

        self.test = test
        if self.test:
            # checkpoints
            self.checkpoints = np.load('checkpoints.npy')
            self.n_checkpoints = self.checkpoints.shape[0]

            phi0 = delta_angle_0 + math.acos(self.checkpoints[0,0]/np.linalg.norm(self.checkpoints[0,:]))

            self.N_hat = self.min_steps(np.array([self.checkpoints[0,0], self.checkpoints[0,1]]),\
                            np.array([0,0]), phi0, np.array([self.checkpoints[1,0], self.checkpoints[1,1]]))
            print('N_hat: ', self.N_hat)
            self.Np = self.N_hat + 1 #prediction horizon
        #else: TODO: add collision detection, lap detaction, tracking of past checkpoints
        
        self.n_constraints = 7*(self.Np-1)+5
        self.model = nmpc_model.nmpc_model(self.Np)

        #IPOPT parameters
        phi0 = delta_angle_0 + math.acos(self.checkpoints[0,0]/np.linalg.norm(self.checkpoints[0,:]))
        self.lb, self.ub, self.cl, self.cu = self.set_limits(x0,y0,phi0)
        

    ########################################################################
    # MAIN interface to csb.py
    # called in every time step
    # given current game state, output desired thrust and heading
    ########################################################################
    def calculate(self, rx, ry, r1x, r1y, delta_angle):
        # TODO: make sure there are no rounding issues
        if self.test:
            checkpointindex = self.get_checkpoint_index(r1x, r1y)
            r1 = self.checkpoints[checkpointindex,:]
            r2 = self.checkpoints[min(checkpointindex + 1, self.n_checkpoints),:]
        else:
            r1 = r2 = np.array([r1x,r1y])

        v = np.array([rx - self.old_x, ry - self.old_y])
        self.old_x = rx
        self.old_y = ry

        self.model.update(np.array([rx,ry]), v, r1, r2, self.N_hat)

        x0 = 10*np.ones((self.Np*7,1))

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
        phi_lim = 2*math.pi
        v_lim = 565 #actual max velocity is 561 in x and y
        #a_min = -100 if test else 0
        a_min = 0
        a_max = 100
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
        cl[5*(Np-1)+1] = r0[1]
        cl[5*(Np-1)+2] = phi0
        cl[5*(Np-1)+3] = v0[0]
        cl[5*(Np-1)+4] = v0[1]
        cu = cl
        #values for v0 constraints are already zero

        x0 = lb

        nlp = ipopt.problem(
            n=7*Np,
            m=n_constraints,
            problem_obj=model,
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu
        )

        nlp.addOption('max_iter', 5000)

        #SOLVE nlp
        x, info = nlp.solve(x0)

        return thrust, next_checkpoint_x, next_checkpoint_y


    ########################################################################
    # UTILITY functions
    ########################################################################
    def get_checkpoint_index(self, checkpoint_x, checkpoint_y):
        for index in range(self.n_checkpoints):
            if self.checkpoints[index,0] == checkpoint_x and self.checkpoints[index,1] == checkpoint_y:
                return index
        return -1

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

    def set_limits(self, x0, y0, phi0):

        x_min = -1000 #all checkpoints in [0,16000]
        x_max = 20000 #all checkpoints in [0,16000]
        y_min = -1000 #all checkpoints in [0,9000]
        y_max = 10000 #all checkpoints in [0,9000]
        phi_lim = 2*math.pi
        v_lim = 15 #actual max velocity is 5.666 in x and y
        #a_min = -100 if self.test else 0
        a_min = 0
        a_max = 100
        w_lim = math.pi/10

        #upper and lower bounds on variables
        #rx,ry,phi,vx,vy,a,w
        lb = np.zeros((self.Np,7))
        ub = np.zeros((self.Np,7))
        for k in range(self.Np):
            lb[k,:] = np.array([x_min, y_min, -phi_lim, -v_lim, -v_lim, a_min, -w_lim])
            ub[k,:] = np.array([x_max, y_max, phi_lim, v_lim, v_lim, a_max, w_lim])
        lb = lb.reshape(7*self.Np,1)
        ub = ub.reshape(7*self.Np,1)

        #upper and lower bounds on constraint functions
        cl = np.zeros((self.n_constraints))
        cu = np.zeros((self.n_constraints))
        for k in range(self.Np-1):
            cl[k] = a_min #TODO: these constraints are possibly redundant
            cu[k] = a_max
            cl[k + 1*(self.Np-1)] = -w_lim
            cu[k + 1*(self.Np-1)] = w_lim

        #values for cu, cl for x,y,vx,vy,psi constraints are already zero
        cl[7*(self.Np-1)] = cu[7*(self.Np-1)] = x0
        cl[7*(self.Np-1)+1] = cu[7*(self.Np-1)+1] = y0
        cl[7*(self.Np-1)+2] = cu[7*(self.Np-1)+2] = phi0
        #values for v0 constraints are already zero
        return lb, ub, cl, cu

    def get_name(self):
        return 'NMPC'
