import math
import numpy as np
from scipy.spatial import distance
import pygame
import game

class csbpod():
    minVel = -600.0
    maxVel = 600.0
    maxThrust = 200.0
    maxSteeringAngle = (
            0.1 * np.pi
    )  # Maximum steering angle in radians [+-0.1 rad = +-15 Deg]
    friction = 0.85
    M_PI2 = 2.0 * np.pi

    def __init__(self, scale, checkpoint, *args, **kwargs):
        self.scale = scale
        self.theta = np.random.randint(0, 359) * np.pi / 180.0
        self.x = checkpoint[0]
        self.y = checkpoint[1]
        self.vx = 0
        self.vy = 0
        self.x_prev = None
        self.y_prev = None
        self.checkpointindex = 0

        # pygame objects
        self.surface = pygame.image.load("img/pod.png")
        self.rect = self.surface.get_rect()
        #start position and centering
        self.rect.x = self.x - 64 
        self.rect.y = self.y - 64

    def getAngle(self, target):
        # Get the angle [0,2*pi] of the vector going from pod's position to a target
        d = np.array([target[0] - self.x, target[1] - self.y])
        norm_d = np.linalg.norm(d)

        angle = math.acos(d[0] / (norm_d + 1e-16))

        if d[1] < 0:
            angle = self.M_PI2 - angle

        return angle

    def getDeltaAngle(self, target):
        # Get the minimum delta angle needed by the pod to reach the target
        angle = self.getAngle(target)

        # Positive amount of angle to reach the target turning to the right of the pod
        right = (
            angle - self.theta
            if self.theta <= angle
            else self.M_PI2 - self.theta + angle
        )
        # Positive amount of angle to reach the target turning to the left of the pod
        left = (
            self.theta - angle
            if self.theta >= angle
            else self.theta + self.M_PI2 - angle
        )

        # Get the minimum delta angle (positive in right, negative in left)
        if right < left:
            return right
        else:
            return -left

    def move(self, targetX, targetY, thrust, game):
        # update angle
        omega = np.clip(self.getDeltaAngle(np.array([targetX, targetY])), -self.maxSteeringAngle, self.maxSteeringAngle)
        self.theta += omega
        if self.theta >= self.M_PI2:
            self.theta -= self.M_PI2
        elif self.theta < 0.0:
            self.theta += self.M_PI2

        # Update dynamics
        self.vx += math.cos(self.theta) * thrust
        self.vy += math.sin(self.theta) * thrust

        self.x = round(self.x + self.vx)
        self.y = round(self.y + self.vy)

        # Apply the friction
        self.vx = int(0.85 * self.vx)
        self.vy = int(0.85 * self.vy)

        self.log('x, y, vx, vy, theta:')
        self.log(str(self.x) + ' ' + str(self.y) + ' ' + str(self.vx) + ' ' + str(self.vy) + ' ' + str(self.theta))
        self.rect.x = self.x/self.scale-64
        self.rect.y = self.y/self.scale-64

        self.check_for_collision(game)


    def check_for_collision(self, game):
        print('checking for collision--------------------------------------------------')
        coordinates = np.array([self.x, self.y])
        checkpoint = game.checkpoints[self.checkpointindex]
        dist = distance.euclidean(coordinates, checkpoint)

        # TODO: detect collision inbetween steps
        if dist < game.checkpointradius:
            self.running = not (self.checkpointindex == (game.n_checkpoints - 1))
            print('checkpoint collision----------------------------------------------')
            self.log('checkpoint collision----------------------------------------------')
            self.log('target, coords, distance = ' + str(checkpoint) + ' ' +  str(coordinates) + ' ' + str(dist))
            self.log('checkpoints remaining: ' + str(game.n_checkpoints - game.checkpointindex - 1))
            self.checkpointindex = game.checkpointindex + 1
            game.checkpointindex = self.checkpointindex
            if game.checkpointindex == game.n_checkpoints:
                game.running = False

    def getState(self, game):
        # check for collision
        if not game.running:
            return  0, 0, 0, 0, 0, 0, 0, 0, game.running
        else:
            coordinates = np.array([self.x, self.y])
            checkpoint = game.checkpoints[self.checkpointindex]
            dist = distance.euclidean(coordinates, checkpoint)

            # get angle between current heading theta and the next checkpoint
            delta_angle = self.getDeltaAngle(np.array([checkpoint[0], checkpoint[1]]))
            return checkpoint[0], checkpoint[1], self.x, self.y, self.theta, self.vx, self.vy, delta_angle, game.running

    def log(self, message):
        filename = 'controller' + '.log'
        logfile =  open(filename, 'w')
        logfile.writelines(message + '\n')
        logfile.close()
