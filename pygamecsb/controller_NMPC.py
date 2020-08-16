import numpy as np
import math
import ipopt
import nmpc_model
from matplotlib import pyplot as plt
import numdifftools as nda

class NMPC():
    ########################################################################
    # INIT for NMPC controller
    # called once before the start of the game
    # given current game state, output desired thrust and heading
    ########################################################################
    def __init__(self, test, x0, y0, delta_angle_0, render_size, scale):
        self.r0 = np.array([x0,y0])
        self.v0 = np.zeros((2))

        self.gamewidth = scale*render_size[0]
        self.gameheight = scale*render_size[1]

        self.Np = 8
        self.Nvar = 7
        self.n_constraints = 5*(self.Np-1) + 5

        self.test = test
        if self.test:
            # checkpoints
            self.checkpoints = np.load('checkpoints.npy')
            self.n_checkpoints = self.checkpoints.shape[0]

            self.phi0 = delta_angle_0 + math.acos(self.checkpoints[0,0]/np.linalg.norm(self.checkpoints[0,:]))
        self.solutions = np.array([])

        self.old_thrust = 0

        self.model = nmpc_model.nmpc_model()
        self.lb, self.ub, self.cl, self.cu = self.bounds_no_inits()
        self.sol = np.zeros((self.Nvar*self.Np))
        self.tick = 0

        
    ########################################################################
    # MAIN interface to csb.py
    # called in every time step
    # given current game state, output desired thrust and heading
    ########################################################################
    def calculate(self, rx, ry, phi, vx, vy, r1x, r1y, delta_angle):
        self.tick += 1
        r0 = np.array([rx,ry])
        v0 = np.array([vx,vy])
        phi0 = phi
        # TODO: make sure there are no rounding issues
        if self.test:
            checkpointindex = self.get_checkpoint_index(r1x, r1y)
            r1 = self.checkpoints[checkpointindex,:]
            # r2 = self.checkpoints[min(checkpointindex + 1, self.n_checkpoints-1),:]
        else:
            r1 = np.array([r1x,r1y])

        self.r0 = np.array([rx, ry])
        self.v0 = np.array([vx, vy])        

        ############ new x0, bounds as inital guess from old prediction 
        # self.Np = min_steps(r0, v0, delta_angle, r1)
        self.model.update_state(r1)
        x0 = self.set_guess()

        cl, cu = self.bounds_inits(r0, phi0, v0)

        nlp = ipopt.problem(
            n=self.Nvar*self.Np,
            m=self.n_constraints,
            problem_obj=self.model,
            lb=self.lb,
            ub=self.ub,
            cl=cl,
            cu=cu
        )

        nlp.addOption('max_iter', 100)

        #SOLVE nlp
        self.sol, info = nlp.solve(x0)
        thrust = self.sol[5]
        w = self.sol[6]

        heading_x, heading_y = self.set_heading(w,phi0)

        print('--------------------------_DONE----------------------------')

        sol = self.sol.reshape(-1,self.Nvar)
        self.plot(sol[:,0], sol[:,1], r0, r1, self.tick)
        # return thrust, heading_x, heading_y
        return thrust, r1x, r1y


    ########################################################################
    # UTILITY functions
    ########################################################################
    def plot(self, rx, ry, r0, r1, tick):
        plt.plot(rx,self.gameheight-ry, 'ko-')
        plt.plot(r0[0],self.gameheight-r0[1], 'go')
        plt.plot(r1[0],self.gameheight-r1[1], 'bo')
        plt.xlim(0,self.gamewidth)
        plt.ylim(0,self.gameheight)

        index = 0
        for x,y in zip(rx,ry):
            label = str(index)
            plt.annotate(label, # this is the text
                         (x,self.gameheight-y), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(0,10), # distance from text to points (x,y)
                         ha='center') # horizontal alignment can be left, right or center
            index += 1

        plt.savefig('fig_' + str(tick) + '.png')
        plt.clf()

    def get_checkpoint_index(self, checkpoint_x, checkpoint_y):
        for index in range(self.n_checkpoints):
            if self.checkpoints[index,0] == checkpoint_x and self.checkpoints[index,1] == checkpoint_y:
                return index
        return -1

    def bounds_no_inits(self):
        x_min = -self.gamewidth/2 
        x_max = self.gameheight*10
        y_min = -self.gamewidth/2
        y_max = self.gamewidth*10
        phi_lim = math.pi
        v_lim = 10000 #actual max velocity is 561 in x and y
        #a_min = -100 if test else 0
        a_min = 0
        a_max = 100
        w_lim = math.pi/10

        #upper and lower bounds on variables
        #rx,ry,phi,vx,vy,a,w
        lb = np.zeros((self.Np,self.Nvar))
        ub = np.zeros((self.Np,self.Nvar))
        for k in range(self.Np):
            lb[k,:] = np.array([x_min, y_min, -phi_lim, -v_lim, -v_lim, a_min, -w_lim])
            ub[k,:] = np.array([x_max, y_max, phi_lim, v_lim, v_lim, a_max, w_lim])
        lb = lb.reshape(self.Nvar*self.Np,1)
        ub = ub.reshape(self.Nvar*self.Np,1)

        #upper and lower bounds on constraint functions
        cl = np.zeros((self.n_constraints))
        cu = np.zeros((self.n_constraints))
        return lb, ub, cl, cu

    def bounds_inits(self, r0, phi0, v0):
        cl = self.cl
        cu = self.cl
        cl[5*(self.Np-1)] = cu[5*(self.Np-1)] =r0[0]
        cl[5*(self.Np-1)+1] = cu[5*(self.Np-1)+1] = r0[1]
        cl[5*(self.Np-1)+2] = cu[5*(self.Np-1)+2] = phi0
        cl[5*(self.Np-1)+3] = cu[5*(self.Np-1)+3] = v0[0]
        cl[5*(self.Np-1)+4] = cu[5*(self.Np-1)+4] = v0[1]
        return cl, cu


    def min_steps(self, r0, v0, deltaphi0, r1):
        x = r0
        v = v0

        d0 = r1 - r0 # distance vector at start point
        dist0 = np.linalg.norm(d0)
        phi0 = math.acos((d0[0]) / dist0) #angle of target at start

        #only need to get rid of vel in wrong direction
        v = v0-np.dot(v0,d0/dist0)*(d0/dist0)

        t_stop = 0
        while np.linalg.norm(v) > 0:
            t_stop += 1
            x = x + v
            v[0] = int(0.85*v[0])
            v[1] = int(0.85*v[1])

        x1 = x #x is now at stop position

        #rotation time
        d1 = r1 - x1 # distance vector at stop point
        dist1 = np.linalg.norm(d1)
        phi1 = math.acos((d1[0]) / dist1) #angle of target at stop

        deltaphi1 = (deltaphi0 + (phi0-phi1))%np.pi
        t_rot = math.ceil(10 * np.abs(deltaphi1) / math.pi) #rotation time at max. +/- pi/10 per tick

        t_travel = 0 #should change condition to sign change maybe? or nondecreasing?
        while np.abs(x[0]-r1[0]) >= 300 and np.abs(x[1]-r1[1]) >= 300:
            t_travel += 1
            x = x + v
            v[0] = int(0.85*v[0] + 85*math.cos(-phi1))
            v[1] = int(0.85*v[1] + 85*math.sin(-phi1))

        print('t_rot,t_stop,t_travel: ', t_rot, t_stop, t_travel)
        return min(max(t_stop, t_rot) + t_travel,8)

    def set_heading(self, w, phi0):
        dx = math.cos(phi0 + w)
        dy = -math.sin(phi0 + w)

        print('heading: ',dx,dy)

        return dx,dy

    def set_guess(self):
        x0 = np.zeros((self.Nvar*self.Np))
        x0[:self.Nvar*(self.Np-1)] = self.sol[self.Nvar:] #remove step already taken
        return x0

    def get_name(self):
        return 'NMPC'
