import math
import numpy as np
from scipy.spatial import distance
import pygame
import datetime


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
        # internal state
        # state= [theta, x, vx, y, vy]
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

        # logging
        time = datetime.datetime.now()
        filename = 'controller_A' + str(time) + '.log'
        self.logfile =  open(filename, 'w')


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

    def move(self, targetX, targetY, thrust):
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
        self.log(str(self.x) + str(self.y) + str(self.vx) + str(self.vy) + str(self.theta))
        self.rect.x = self.x/self.scale
        self.rect.y = self.y/self.scale

    def getState(self, game, running):
        # check for collision
        coordinates = np.array([self.x, self.y])
        checkpoint = game.checkpoints[self.checkpointindex]
        dist = distance.euclidean(coordinates, checkpoint)
        if dist < game.checkpointradius:
            running = not (self.checkpointindex == (game.n_checkpoints - 1))
            self.log('checkpoint collision----------------------------------------------')
            self.log('target, coords, distance = ' + str(checkpoint) + str(coordinates) + str(dist))
            self.log('checkpoints remaining: ' + str(game.n_checkpoints - game.checkpointindex - 1))
            self.checkpointindex = game.checkpointindex + 1
            game.checkpointindex = self.checkpointindex
        return checkpoint[0], checkpoint[1], self.x, self.y, self.vx, self.vy, running

    def log(self, message):
        self.logfile.write(message)

class game:
    # Global game parameters
    def __init__(self, width, height, n_checkpoints, scale):
        self.checkpointindex = 0
        self.scale = scale
        self.gameWidth = width*self.scale
        self.gameHeight = height*self.scale
        self.n_checkpoints = n_checkpoints
        self.checkpointradius = self.gameHeight/70
        self.checkpoints = np.zeros((self.n_checkpoints, 2))

        self.checkpointSurface = pygame.image.load("img/ckpt.png")
        # list of checkpoint rectangles
        self.checkpoints = self.genCheckpoints(n_checkpoints)

    def genCheckpoints(self, n):
        # checkpoints is array with coords of all checkpoints
        checkpoints = np.zeros((n, 2))
        for index in range(n):
            while True:
                tooclose = False
                # sample five checkpoints with a minimum distance between them
                ckpt = np.array([np.random.randint(0, 0.9 * self.gameWidth),
                                 np.random.randint(0, 0.9 * self.gameHeight)])
                for i in range(index):
                    if distance.euclidean(ckpt, self.checkpoints[i - 1, :]) <= 450:
                        tooclose = True
                if not tooclose:
                    checkpoints[index, :] = ckpt
                    break
        np.save('checkpoints', checkpoints)
        return checkpoints

    def CheckpointRect(self, checkpoint):
        # create n rects and move to coordinates
        rect = self.checkpointSurface.get_rect()
        rect.x += (checkpoint[0] - 45)/self.scale
        rect.y += (checkpoint[1] - +45)/self.scale
        return rect
