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
render = True
########################################################################
# set GAME PARAMETERS here
########################################################################
scale = 10 # game size = renderSize*scale
renderSize = renderWidth, renderHeight = 1600, 900
n_checkpoints = 5
seed = int(sys.argv[1])
np.random.seed(seed)

game = game.game(renderWidth, renderHeight, n_checkpoints, scale)
theta0 = np.random.randint(0, 359) * np.pi / 180.0
pod1 = pod.csbpod(scale, game.checkpoints[0,:], 1, theta0)
pod2 = pod.csbpod(scale, game.checkpoints[0,:], 2, theta0)
trajectory = np.array([])
if render:
    screen = pygame.display.set_mode(renderSize)
    background = pygame.image.load("img/back.png")
    background = pygame.transform.scale(background, (renderWidth, renderHeight))
running = True

########################################################################
# get INITIAL CONDITIONS
target_x,target_y,x,y,theta,vx,vy,delta_angle,running=pod1.getState(game)
# target2_x,target2_y,x2,y2,theta2,vx2,vy2,delta_angle2,running2=pod2.getState(game)
########################################################################

# controller = int(sys.argv[2])
# if controller == 0:
#     control = controller_PID.PID()
# else:
#     control = controller_NMPC.NMPC(test, x, y, delta_angle, renderSize, scale)
########################################################################
control = controller_PID.PID()
control2 = controller_NMPC.NMPC(test, x, y, theta, renderSize, scale)

tick = 0
while running or running2:
    tick +=1
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

    target_x, target_y, x, y, theta, vx, vy, delta_angle, running = pod1.getState(game)
    target2_x, target2_y, x2, y2, theta2, vx2, vy2, delta_angle2, running2 = pod2.getState(game)

    # delta_angle is the angle between the current heading and the target
    thrust, heading_x, heading_y = control.calculate(x, y, theta, vx, vy, target_x, target_y, delta_angle)
    thrust2, heading2_x, heading2_y = control2.calculate(x2, y2, theta2, vx2, vy2, target2_x, target2_y, delta_angle2)

    if not test:
        trust = np.clip(0,100, thrust)

    pod1.move(heading_x, heading_y, thrust, game)
    pod2.move(heading2_x, heading2_y, thrust2, game)

    if render:
        screen.blit(background, (0, 0))
        for checkpoint in game.checkpoints:
            rect = game.checkpointRect(checkpoint)
            screen.blit(game.checkpointSurface, rect)
        pod1.surface = pygame.image.load("img/pod.png")
        pod1.surface = pygame.transform.rotate(pod1.surface, -pod1.theta*180/np.pi)
        screen.blit(pod1.surface, pod1.rect)
        pod2.surface = pygame.image.load("img/pod-2.png")
        pod2.surface = pygame.transform.rotate(pod2.surface, -pod2.theta*180/np.pi)
        screen.blit(pod2.surface, pod2.rect)
        pygame.display.flip()
    # trajectory = np.append(trajectory,np.array([x,y,thrust,theta]))
    if tick >= 100:
        break
# target_x, target_y, x, y, theta, vx, vy, delta_angle, running = pod.getState(game)
# trajectory = np.append(trajectory,np.array([x,y,thrust,theta]))

# np.save('tmp/'+control.get_name()+'_'+str(seed), trajectory)
# np.save('tmp/checkpoints_' + str(seed), game.checkpoints)
# filename = 'tmp/score_'+control.get_name()+'_'+str(seed)
# logfile = open(filename, 'a')
# logfile.writelines('controller ' + control.get_name() + ':\n')
# logfile.writelines('reached target in ' + str(tick) + ' ticks' + '\n')
# logfile.writelines(str(tick) + '\n')
# logfile.writelines('n_checkpoints: ' + str(n_checkpoints-1) + '\n')
# logfile.writelines('random seed: ' + str(seed) + '\n')
# logfile.close()
pygame.quit()
