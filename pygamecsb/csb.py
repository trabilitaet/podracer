import sys, pygame
import pod
import numpy as np
import datetime
#import controller_A
import controller_PID
# import controller_NMPC
########################################################################
# enable TEST MODE here:
test = True
# test mode has no limits on thrust
# -> thrust can be negative
# test mode assumes all checkpoints 
# known and exported to file
########################################################################

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
# initialize controller
#control = controller_A.controller_A()
control = controller_PID.PID()
# control = controller_NMPC.NMPC(test, x, y, delta_angle)
########################################################################

tick = 0
running = True
while running:
    tick +=1
    print('tick: ', tick)
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

    target_x, target_y, x, y, vx, vy, delta_angle, running = pod.getState(game, running)
    ################################
    # controller goes here here
    ################################
    # delta_angle is the angle between the current heading and the target
    thrust, heading_x, heading_y = control.calculate(x, y, target_x, target_y, delta_angle)
    
    ################################
    # end controller here here
    ################################

    if not test:
        trust = np.clip(0,100, thrust)

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
filename = 'score' + control.getName()
logfile = open(filename, 'w')
logfile.writelines('reached target in ' + str(tick) + ' ticks' + '\n')
logfile.close()
