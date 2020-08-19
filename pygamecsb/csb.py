import sys, pygame
import pod
import game
import numpy as np
import controller_A
import controller_PID
import controller_NMPC
########################################################################
# enable TEST MODE here:
test = True
# test mode has NO LIMIT ON THRUST
# -> thrust can be negative
# test mode assumes ALL CHECKPOINTS KNOWN
# -> exported to file
########################################################################

pygame.init()

########################################################################
# set GAME PARAMETERS here
########################################################################
scale = 10 # game size = renderSize*scale
renderSize = renderWidth, renderHeight = 1600, 900
n_checkpoints = 5
# np.random.seed(1518)
seed = 518
np.random.seed(seed)

screen = pygame.display.set_mode(renderSize)
game = game.game(renderWidth, renderHeight, n_checkpoints, scale)
pod = pod.csbpod(scale, game.checkpoints[0,:])
background = pygame.image.load("img/back.png")
background = pygame.transform.scale(background, (renderWidth, renderHeight))
running = True

########################################################################
# get INITIAL CONDITIONS
target_x,target_y,x,y,theta,vx,vy,delta_angle,running=pod.getState(game)
########################################################################

########################################################################
# initialize CONTROLLER
#control = controller_A.controller_A()
control = controller_PID.PID()
# control = controller_NMPC.NMPC(test, x, y, delta_angle, renderSize, scale)
########################################################################

tick = 0
while running:
    tick +=1
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

    target_x, target_y, x, y, theta, vx, vy, delta_angle, running = pod.getState(game)

    # delta_angle is the angle between the current heading and the target
    thrust, heading_x, heading_y = control.calculate(x, y, theta, vx, vy, target_x, target_y, delta_angle)

    if not test:
        trust = np.clip(0,100, thrust)

    pod.move(heading_x, heading_y, thrust, game)
    # render game
    screen.blit(background, (0, 0))
    for checkpoint in game.checkpoints:
        rect = game.checkpointRect(checkpoint)
        screen.blit(game.checkpointSurface, rect)
    pod.surface = pygame.image.load("img/pod.png")
    pod.surface = pygame.transform.rotate(pod.surface, -pod.theta*180/np.pi)
    screen.blit(pod.surface, pod.rect)
    pygame.display.flip()

filename = 'score' + control.get_name()
logfile = open(filename, 'a')
logfile.writelines('controller ' + control.get_name() + ':\n')
logfile.writelines('reached target in ' + str(tick) + ' ticks' + '\n')
logfile.writelines('n_checkpoints: ' + str(n_checkpoints) + '\n')
logfile.writelines('random seed: ' + str(seed) + '\n')
logfile.close()
pygame.quit()
