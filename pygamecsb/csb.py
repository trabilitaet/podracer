import sys, pygame
import pod
import numpy as np

pygame.init()

gameSize = gameWidth, gameHeight = 1600, 900
n_checkpoints = 6
np.random.seed(1681)

screen = pygame.display.set_mode(gameSize)
game = pod.game(gameWidth, gameHeight, n_checkpoints)
pod = pod.csbpod()
background = pygame.image.load("img/back.png")
background = pygame.transform.scale(background, (gameWidth, gameHeight))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()
    target_x, target_y, x, y, vx, vy, running = pod.getState(game, running)

    ################################
    # calculate heading, thrust here
    ################################



    heading_x, heading_y = target_x, target_y
    thrust = 3
    ################################
    # end controller here here
    ################################

    # move pod
    pod.move(heading_x, heading_y, thrust)
    # render game
    screen.blit(background, (0, 0))
    for checkpoint in game.checkpoints:
        rect = game.CheckpointRect(checkpoint)
        screen.blit(game.checkpointSurface, rect)
    # rotate pod appropriately
    pod.surface = pygame.image.load("img/pod.png")
    pod.surface = pygame.transform.rotate(pod.surface, -pod.theta*180/np.pi)
    screen.blit(pod.surface, pod.rect)
    pygame.display.flip()
