import sys, pygame
import pod
import game
import numpy as np
import controller_A
#import controller_A
import controller_PID
import controller_NMPC
########################################################################
# enable TEST MODE here:
test = False
# test mode has no limits on thrust
# -> thrust can be negative
# test mode assumes all checkpoints 
# known and exported to file
########################################################################

pygame.init()

########################################################################
# set GAME PARAMETERS here
########################################################################
scale = 10
gameSize = gameWidth, gameHeight = 1600, 900
# gameSize = gameWidth, gameHeight = 800, 450
n_checkpoints = 2
np.random.seed(117)

screen = pygame.display.set_mode(gameSize)
game = game.game(gameWidth, gameHeight, n_checkpoints, scale)
pod = pod.csbpod(scale, game.checkpoints[0,:])
background = pygame.image.load("img/back.png")
background = pygame.transform.scale(background, (gameWidth, gameHeight))
running = True

########################################################################
# get INITIAL CONDITIONS
target_x,target_y,x,y,vx,vy,delta_angle,running=pod.getState(game, running)
########################################################################

########################################################################
# initialize CONTROLLER
#control = controller_A.controller_A()
#control = controller_PID.PID()
control = controller_NMPC.NMPC(test, x, y, delta_angle, gameSize)
########################################################################

tick = 0
while running:
    tick +=1
    print('tick: ', tick)
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

    target_x, target_y, x, y, vx, vy, delta_angle, running = pod.getState(game, running)

    # delta_angle is the angle between the current heading and the target
    thrust, heading_x, heading_y = control.calculate(x, y, vx, vy, target_x, target_y, delta_angle)

    if not test:
        trust = np.clip(0,100, thrust)

    print('r1: ', target_x, target_y)

    # move pod
    pod.move(heading_x, heading_y, thrust)
    # render game
    screen.blit(background, (0, 0))
    for checkpoint in game.checkpoints:
        rect = game.checkpointRect(checkpoint)
        screen.blit(game.checkpointSurface, rect)
    # rotate pod (reload to make it 0 first)
    pod.surface = pygame.image.load("img/pod.png")
    pod.surface = pygame.transform.rotate(pod.surface, -pod.theta*180/np.pi)
    screen.blit(pod.surface, pod.rect)
    pygame.display.flip()

filename = 'score' + control.get_name()
logfile = open(filename, 'w')
logfile.writelines('reached target in ' + str(tick) + ' ticks' + '\n')
logfile.close()
pygame.quit()
