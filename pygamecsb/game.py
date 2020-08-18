import math
import numpy as np
from scipy.spatial import distance
import pygame
import pod

class game:
    # Global game parameters
    def __init__(self, width, height, n_checkpoints, scale):
        self.checkpointindex = 0
        self.scale = scale
        self.gamewidth = width*self.scale
        self.gameheight = height*self.scale
        self.n_checkpoints = n_checkpoints
        self.checkpointradius = int(self.gameheight/(2*self.scale))
        print('checkpointradius: ', self.checkpointradius)
        self.checkpoints = np.zeros((self.n_checkpoints, 2))

        self.checkpointSurface = pygame.image.load("img/ckpt.png")
        self.checkpoints = self.genCheckpoints(n_checkpoints)
        self.running = True

    def genCheckpoints(self, n):
        # checkpoints is array with coords of all checkpoints
        checkpoints = np.zeros((n, 2))
        for index in range(n):
            while True:
                tooclose = False
                # sample five checkpoints with a minimum distance between them
                ckpt = np.array([np.random.randint(0, 0.85 * self.gamewidth),
                                 np.random.randint(0, 0.85 * self.gameheight)])
                for i in range(index):
                    if distance.euclidean(ckpt, self.checkpoints[i - 1, :]) <= 450*self.scale:
                        tooclose = True
                if not tooclose:
                    checkpoints[index, :] = ckpt
                    break
        np.savetxt('checkpoints.txt', checkpoints)
        np.save('checkpoints', checkpoints)
        return checkpoints

    def checkpointRect(self, checkpoint):
        # create n rects and move to coordinates
        rect = self.checkpointSurface.get_rect()
        rect.x += (checkpoint[0])/self.scale - 45
        rect.y += (checkpoint[1])/self.scale - 45
        return rect
