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
    def __init__(self, test, x0, y0, delta_angle_0, gamesize, scale):
        self.r0 = np.array([x0,y0])
        self.v0 = np.zeros((2))

        self.gamewidth = scale*gamesize[0]
        self.gameheight = scale*gamesize[1]

        self.test = test
        if self.test:
            # checkpoints
            self.checkpoints = np.load('checkpoints.npy')
            self.n_checkpoints = self.checkpoints.shape[0]

            self.phi0 = delta_angle_0 + math.acos(self.checkpoints[0,0]/np.linalg.norm(self.checkpoints[0,:]))
        self.solutions = np.array([])

        self.old_thrust = 0
        self.calculated = False
        
    ########################################################################
    # MAIN interface to csb.py
    # called in every time step
    # given current game state, output desired thrust and heading
    ########################################################################
    def calculate(self, rx, ry, vx, vy, r1x, r1y, delta_angle):
        # plt.plot(rx,ry, 'rx-')

        if self.calculated:
            if self.solutions.shape[0] > 0:
                thrust = self.solutions[0]
                w = self.solutions[1]
                
                self.solutions = self.solutions[2:]
                heading_x, heading_y = self.set_heading(w,np.array([rx,ry]),np.array([r1x,r1y]),delta_angle)
                return thrust, heading_x, heading_y


        # TODO: make sure there are no rounding issues
        if self.test:
            checkpointindex = self.get_checkpoint_index(r1x, r1y)
            r1 = self.checkpoints[checkpointindex,:]
            r2 = self.checkpoints[min(checkpointindex + 1, self.n_checkpoints-1),:]
        else:
            r1 = r2 = np.array([r1x,r1y])

        self.r0 = np.array([rx, ry])
        self.v0 = np.array([vx, vy])
        phi0 = delta_angle + math.acos((r1[0]-self.r0[0])/np.linalg.norm(r1-self.r0))
        print('r0,v0,r1: ', self.r0, self.v0,r1)


        self.Np = self.min_steps(self.r0, self.v0, delta_angle, r1)
        # self.Np = 5
        print('Np: ',self.Np)
        self.n_constraints = 5*(self.Np-1) + 5

        model = nmpc_model.nmpc_model(self.r0,self.v0,r1,self.Np)
        lb, ub, cl, cu, x0 = self.bounds(self.r0,r1,phi0,self.v0, self.Np)


        nlp = ipopt.problem(
            n=7*self.Np,
            m=self.n_constraints,
            problem_obj=model,
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu
        )

        nlp.addOption('max_iter', 100)

        #SOLVE nlp
        x, info = nlp.solve(x0)
        self.calculated = True
        for k in range(1,self.Np-1):
            self.solutions = np.append(self.solutions, x[5+k*7])
            self.solutions = np.append(self.solutions, x[6+k*7])
        thrust = x[5]
        w = x[6]

        heading_x, heading_y = self.set_heading(w,self.r0,r1,delta_angle)

        print('--------------------------_DONE----------------------------')
        # print('thrust, w, heading_x, heading_y: ', thrust, w, heading_x, heading_y)


        sol = x.reshape(-1,7)
        rx,ry,phi,vx,vy,a,w = sol[:,0],sol[:,1],sol[:,2],sol[:,3],sol[:,4],sol[:,5],sol[:,6]

        self.plot(rx,ry,self.r0,r1)

        self.old_thrust = thrust
        # return thrust, heading_x, heading_y
        return thrust, r1x, r1y


    ########################################################################
    # UTILITY functions
    ########################################################################
    def plot(self,rx,ry,r0,r1):
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

        plt.show()

    def get_checkpoint_index(self, checkpoint_x, checkpoint_y):
        for index in range(self.n_checkpoints):
            if self.checkpoints[index,0] == checkpoint_x and self.checkpoints[index,1] == checkpoint_y:
                return index
        return -1

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

    def bounds(self,r0,r1,phi0,v0,Np):
        Np = int(Np)

        x_min = -self.gamewidth/2 
        x_max = self.gameheight*10
        y_min = -self.gamewidth/2
        y_max = self.gamewidth*10
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
        x0 = 0.1*np.ones((Np,7))
        for k in range(Np):
            lb[k,:] = np.array([x_min, y_min, -phi_lim, -v_lim, -v_lim, a_min, -w_lim])
            ub[k,:] = np.array([x_max, y_max, phi_lim, v_lim, v_lim, a_max, w_lim])
            # x0[k,0] = r0[0]
            # x0[k,1] = r0[1]
        lb = lb.reshape(7*Np,1)
        ub = ub.reshape(7*Np,1)
        x0 = x0.reshape(7*Np,1)

        #upper and lower bounds on constraint functions
        cl = np.zeros((self.n_constraints))
        cu = np.zeros((self.n_constraints))

        #values for cu, cl for x,y,vx,vy,psi constraints are already zero
        cl[5*(Np-1)] = cu[5*(Np-1)] =r0[0]
        cl[5*(Np-1)+1] = cu[5*(Np-1)+1] = r0[1]
        cl[5*(Np-1)+2] = cu[5*(Np-1)+2] = phi0
        cl[5*(Np-1)+3] = cu[5*(Np-1)+3] = v0[0]
        cl[5*(Np-1)+4] = cu[5*(Np-1)+4] = v0[1]

        return lb, ub, cl, cu, x0

    def set_heading(self,w,r0,r1,delta_angle):
        print(w,r0,r1,delta_angle)
        d = r1-r0
        norm_d = np.linalg.norm(d)

        angle = math.acos(d[0] / (norm_d + 1e-16))

        if d[1] < 0:
            angle = 2*np.pi - angle

        dx = math.cos(angle + delta_angle + w)
        dy = -math.sin(angle + delta_angle + w)

        print('target vs heading: ', d, dx,dy)

        return dx,dy

    def get_name(self):
        return 'NMPC'
