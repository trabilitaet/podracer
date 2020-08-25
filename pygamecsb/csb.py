import sys, pygame
import pod
import game
import numpy as np
from matplotlib import pyplot as plt
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
# turn RENDERING on/off here
########################################################################
render = False
########################################################################
# set GAME PARAMETERS here
########################################################################
scale = 10 # game size = renderSize*scale
renderSize = renderWidth, renderHeight = 1600, 900
n_checkpoints = 4
seed = int(sys.argv[1])
np.random.seed(seed)

game = game.game(renderWidth, renderHeight, n_checkpoints, scale)
pod = pod.csbpod(scale, game.checkpoints[0,:])
trajectory = np.array([])
if render:
    screen = pygame.display.set_mode(renderSize)
    background = pygame.image.load("img/back.png")
    background = pygame.transform.scale(background, (renderWidth, renderHeight))
running = True

########################################################################
# get INITIAL CONDITIONS
target_x,target_y,x,y,theta,vx,vy,delta_angle,running=pod.getState(game)
########################################################################

controller = int(sys.argv[2])
if controller == 0:
    control = controller_PID.PID()
else:
    control = controller_NMPC.NMPC(test, x, y, delta_angle, renderSize, scale)
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

    if render:
        screen.blit(background, (0, 0))
        for checkpoint in game.checkpoints:
            rect = game.checkpointRect(checkpoint)
            screen.blit(game.checkpointSurface, rect)
        pod.surface = pygame.image.load("img/pod.png")
        pod.surface = pygame.transform.rotate(pod.surface, -pod.theta*180/np.pi)
        screen.blit(pod.surface, pod.rect)
        pygame.display.flip()
    trajectory = np.append(trajectory,np.array([x,y,thrust,theta]))
trajectory = np.append(trajectory,np.array([x,y,thrust,theta]))

# np.save('tmp/'+control.get_name()+'_'+str(seed), trajectory)
# np.save('tmp/checkpoints_' + str(seed), game.checkpoints)
filename = 'tmp/score_'+control.get_name()+'_'+str(seed)
logfile = open(filename, 'a')
# logfile.writelines('controller ' + control.get_name() + ':\n')
# logfile.writelines('reached target in ' + str(tick) + ' ticks' + '\n')
logfile.writelines(str(tick) + '\n')
# logfile.writelines('n_checkpoints: ' + str(n_checkpoints-1) + '\n')
# logfile.writelines('random seed: ' + str(seed) + '\n')
logfile.close()
pygame.quit()
