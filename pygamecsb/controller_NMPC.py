import numpy as np
import math
import ipopt
import nmpc_model
from matplotlib import pyplot as plt
import numdifftools as nda

plot_results = True

class NMPC():
    ########################################################################
    # INIT for NMPC controller
    # called once before the start of the game
    # given current game state, output desired thrust and heading
    ########################################################################
    def __init__(self, test, x0, y0, delta_angle_0, render_size, scale):
        self.r0 = np.array([x0,y0])
        self.r1 = np.array([x0,y0])
        self.r2 = np.array([x0,y0])
        self.v0 = np.zeros((2))

        self.gamewidth = scale*render_size[0]
        self.gameheight = scale*render_size[1]

        self.Np = 10
        self.Nvar = 7
        self.n_constraints = 5*(self.Np-1) + 5

        self.checkpointindex = 1 # 0 is starting position
        self.checkpointradius = int(self.gameheight/(2*scale))
        self.test = test
        if self.test:
            # checkpoints
            self.checkpoints = np.load('checkpoints.npy')
            self.n_checkpoints = self.checkpoints.shape[0]

            self.phi0 = delta_angle_0 + math.acos(self.checkpoints[0,0]/np.linalg.norm(self.checkpoints[0,:]))


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
            self.checkpointindex = self.get_checkpoint_index(r1x, r1y)
            r1 = self.checkpoints[self.checkpointindex,:]
            r2 = self.checkpoints[min(self.checkpointindex + 1, self.n_checkpoints-1),:]
        else:
            r1 = np.array([r1x,r1y])
            r2 = np.array([r1x,r1y]) # avoid passing by reference        

        self.model.update_state(r1,r2)
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

        nlp.addOption('max_iter', 500)

        #SOLVE nlp
        self.sol, info = nlp.solve(x0)
        thrust = self.sol[5]
        w = self.sol[6]

        heading_x, heading_y = self.set_heading(w,phi0)

        print('-----------------------OPT_DONE-------------------------')

        sol = self.sol.reshape(-1,self.Nvar)
        self.check_objective(sol)
        if plot_results:
            self.plot(sol, r0, r1, self.tick)
        return thrust, r1x, r1y


    ########################################################################
    # UTILITY functions
    ########################################################################
    def plot(self, sol, r0, r1, tick):
        rx,ry,phi,vx,vy,a,w=sol[:,0],sol[:,1],sol[:,2],sol[:,3],sol[:,4],sol[:,5],sol[:,6]
        plt.subplot(6,1,1)
        plt.xlabel('rx')
        plt.ylabel('ry')
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

        plt.subplot(6,1,2)
        plt.ylabel('phi')
        plt.plot(phi, 'ko-')

        plt.subplot(6,1,3)
        plt.ylabel('vx')
        plt.plot(vx, 'ko-')

        plt.subplot(6,1,4)
        plt.ylabel('vy')
        plt.plot(vy, 'ko-')

        plt.subplot(6,1,5)
        plt.ylabel('a')
        plt.plot(a, 'ko-')

        plt.subplot(6,1,6)
        plt.ylabel('w')
        plt.plot(w, 'ko-')

        plt.savefig('run/fig_' + str(tick) + '.png')
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
        phi_lim = 6*math.pi
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

    def set_heading(self, w, phi0):
        dx = math.cos(phi0 + w)
        dy = -math.sin(phi0 + w)
        return dx,dy

    def set_guess(self):
        x0 = np.zeros((self.Nvar*self.Np))
        x0[:self.Nvar*(self.Np-1)] = self.sol[self.Nvar:] #remove step already taken
        return x0

    def check_objective(self, sol):
        # check if checkpoint can be reached in current horizon
        if np.linalg.norm(self.r1-self.r2) >= 0: #next checkpoint unknown
            return

        rx, ry = sol[:,0], sol[:,1]
        index = 0
        for k in range(self.Np):
            index += 1
            dist = np.linalg.norm(np.array([rx,ry]),self.r1)
            if dist <= self.checkpointradius:
                if index == 0:
                    self.model.set_N_hat(self.Np)
                else:
                    self.model.set_N_hat(index)
        return

    def get_name(self):
        return 'NMPC'
