import numpy as np
import math
import ipopt
import nmpc_model

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

            self.N_hat = self.min_steps(np.array([self.checkpoints[0,0], self.checkpoints[0,1]]),\
                            np.array([0,0]), 0, np.array([self.checkpoints[1,0], self.checkpoints[1,1]]))
            self.Np = self.N_hat + 1 #prediction horizon
        #else: TODO: add collision detection, lap detaction, tracking of past checkpoints
        
        self.n_constraints = 7*(self.Np-1)+5
        self.model = nmpc_model.nmpc_model()

        #IPOPT parameters
        phi0 = delta_angle_0 + math.acos(self.checkpoints[0,0]/np.linalg.norm(self.checkpoints[0,:]))
        self.lb, self.ub, self.cl, self.cu = self.set_limits(x0,y0,phi0)
        

    ########################################################################
    # MAIN interface to csb.py
    # called in every time step
    # given current game state, output desired thrust and heading
    ########################################################################
    def calculate(self, rx, ry, r1x, r1y, delta_angle):
        x = np.array([rx,ry])
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

        self.N_hat = max(1, self.N_hat-1) #TODO
        self.model.update(x, v, r1, r2, self.N_hat)

        x0 = np.ones((self.Np,7))

        nlp = ipopt.problem(
            n=7*self.Np,
            m=len(self.cl),
            problem_obj=self.model,
            lb=self.lb,
            ub=self.ub,
            cl=self.cl,
            cu=self.cu
        )

        #SOLVE nlp
        x, info = nlp.solve(x0)
        print(info)
        plt.imshow(x)
        plt.show()

        return thrust, next_checkpoint_x, next_checkpoint_y


    ########################################################################
    # UTILITY functions
    ########################################################################
    def get_checkpoint_index(self, checkpoint_x, checkpoint_y):
        for index in range(self.n_checkpoints):
            if self.checkpoints[index,0] == checkpoint_x and self.checkpoints[index,1] == checkpoint_y:
                return index
        return -1

    def min_steps(self,x0, v0, phi0, r1):
        #velocity "decay" time
        if np.linalg.norm(v0):
            t_stop = math.ceil(math.log(np.linalg.norm(v0)) / math.log(20 / 17))
        else:
            t_stop = 0
        #print('t_stop: ', t_stop)

        # position as velocity reaches 0
        x1 = x0
        for i in range(t_stop):
            x1 = x1 + pow(17/20,i)*v0

        d = r1 - x1 # distance vector
        dist = np.linalg.norm(d)

        angle = math.acos((d[0]) / dist) #angle to target
        #print('angle: ', angle)
        t_rot = math.ceil(10 * angle / math.pi) #rotation time at max. +/- pi/10 per tick
        #print('t_rot: ', t_rot)

        # calculate travel time from x1 to r1 at max acelleration (formula from mathematica)
        t_travel = math.ceil(17 / 3 + 3 * dist / 172)
        #print('t_travel: ', t_travel)

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
        lb = [x_min, y_min, -phi_lim, -v_lim, -v_lim, a_min, -w_lim]
        ub = [x_max, y_max, phi_lim, v_lim, v_lim, a_max, w_lim]

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
