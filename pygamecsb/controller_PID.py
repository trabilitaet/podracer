import sys
import math
import numpy as np
from scipy.spatial import distance


class PID():
    def __init__(self):
        self.Kp = 0.02
        self.Ki = 0.0000001
        self.Kd = 0.00002

        self.max_thrust = 100
        self.time = 0
        self.error = 0
        self.cum_error = 0
        self.old_error = 0

    def calculate(self, x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_angle):
        self.time += 1

        self.old_error = self.error
        self.error = distance.euclidean(np.array([x,y]), np.array([next_checkpoint_x, next_checkpoint_y]))
        self.cum_error += self.error * self.time
        self.rate_error = (self.error - self.old_error) / self.time

        pid = self.Kp * self.error + self.Ki * self.cum_error + self.Kd * self.rate_error
        #thrust = int(pid - abs(next_checkpoint_angle))
        thrust = int(pid)
        print("thrust: " + str(thrust), file=sys.stderr)

        thrust = np.clip(thrust,0, 100)

        return thrust, next_checkpoint_x, next_checkpoint_y

    def getName(self):
        return 'PID'