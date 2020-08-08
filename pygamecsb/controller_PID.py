import sys
import math
import numpy as np
from scipy.spatial import distance


class PID():
    def __init__(self):
        self.Kp = 0.02
        self.Ki = 0.000000
        self.Kd = 0.003

        self.max_thrust = 100
        self.min_thrust = 0
        self.time = 0
        self.los_error = 0
        self.cum_los_error = 0
        self.old_los_error = 0


    def calculate(self, x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_angle):
        self.time += 1

        print(next_checkpoint_angle)
        dist = np.linalg.norm(np.array([x,y])-np.array([next_checkpoint_x,next_checkpoint_y]))
        self.old_los_error = self.los_error
        self.los_error = math.cos(next_checkpoint_angle)*dist
        print(self.los_error)
        self.cum_los_error += self.los_error * self.time
        print(self.cum_los_error)
        self.rate_los_error = (self.los_error - self.old_los_error) / self.time
        print(self.rate_los_error)

        pid = self.Kp * self.los_error + self.Ki * self.cum_los_error + self.Kd * self.rate_los_error
        thrust = int(pid)
        thrust = np.clip(thrust,0, 100)

        print("thrust: " + str(thrust))
        print("dist: ", )
        return thrust, next_checkpoint_x, next_checkpoint_y        

    def get_name(self):
        return 'PID'