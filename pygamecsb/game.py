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
                    if distance.euclidean(ckpt, self.checkpoints[i - 1, :]) <= 450/self.scale:
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
        rect.x += (checkpoint[0] - 45)/self.scale
        rect.y += (checkpoint[1] - 45)/self.scale
        return rect
