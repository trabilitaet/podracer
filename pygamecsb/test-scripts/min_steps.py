import numpy as np
import math
from matplotlib import pyplot as plt


x0 = np.array([100,100])
# vmax = 561 in both x and y
v0 = np.array([10,-15])
deltaphi0 = 2*np.pi/3
r1 = np.array([2,2])
r2 = np.array([2,2])

#minsteps
v = v0
x = x0

t_stop = 0
while np.linalg.norm(v) > 0:
	t_stop += 1
	x = x + v
	v[0] = int(0.85*v[0])
	v[1] = int(0.85*v[1])

#x is now at stop position
x1 = x
print('x1: ', x1)
print('r1: ', r1)
#rotation time
d0 = r1 - x0 # distance vector at start point
dist0 = np.linalg.norm(d0)
d1 = r1 - x1 # distance vector at stop point
print('d1: ', d1)
dist1 = np.linalg.norm(d1)
print('dist: ', dist1)

phi0 = math.acos((d0[0]) / dist0) #angle of target at start
phi1 = math.acos((d1[0]) / dist1) #angle of target at stop
print('phi1: ', phi1)

deltaphi1 = (deltaphi0 + (phi0-phi1))%2*np.pi
t_rot = math.ceil(10 * np.abs(deltaphi1) / math.pi) #rotation time at max. +/- pi/10 per tick
print('t_rot: ', t_rot)

t_travel = 0
while np.abs(x[0]-r1[0]) <= 300 and np.abs(x[1]-r1[1]) <= 300:
	print(x, v)
	t_travel += 1
	x = x + v
	v[0] = int(0.85*v[0] + 85*math.cos(-phi1))
	v[1] = int(0.85*v[1] + 85*math.sin(-phi1))
print('t_travel: ', t_travel)


