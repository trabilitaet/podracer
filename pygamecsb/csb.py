import sys, pygame
import pod
import numpy as np
import datetime

pygame.init()

scale = 10
gameSize = gameWidth, gameHeight = 1600, 900
n_checkpoints = 6
np.random.seed(1681)

screen = pygame.display.set_mode(gameSize)
game = pod.game(gameWidth, gameHeight, n_checkpoints, scale)
pod = pod.csbpod(scale, game.checkpoints[0,:])
background = pygame.image.load("img/back.png")
background = pygame.transform.scale(background, (gameWidth, gameHeight))

####################################
# enable test mode here:
test = True
# test mode has no limits on thrust
# -> thrust can be negative
####################################

tick = 0
running = True
while running:
    tick +=1
    print('tick: ', tick)
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

    target_x, target_y, x, y, vx, vy, running = pod.getState(game, running)
    ################################
    # controller goes here here
    ################################






    heading_x, heading_y = target_x, target_y
    thrust = 20
    ################################
    # end controller here here
    ################################

    if not test:
        trust = max(0, min(100, thrust))

    # move pod
    pod.move(heading_x, heading_y, thrust)
    # render game
    screen.blit(background, (0, 0))
    for checkpoint in game.checkpoints:
        rect = game.CheckpointRect(checkpoint)
        screen.blit(game.checkpointSurface, rect)
    # rotate pod appropriately (reload to make it 0 first)
    pod.surface = pygame.image.load("img/pod.png")
    pod.surface = pygame.transform.rotate(pod.surface, -pod.theta*180/np.pi)
    screen.blit(pod.surface, pod.rect)
    pygame.display.flip()

pygame.quit()
time = datetime.datetime.utcnow()
filename = 'score' + str(time) + control.getName()
logfile = open(filename, 'w')
logfile.writelines('reached target in ' + str(tick) + ' ticks')
logfile.writelines('')
logfile.close()
