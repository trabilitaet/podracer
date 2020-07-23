import math
import numpy as np

### idea: try controller 4.2
## consider control input tau_u forward accelleration
## x =  [z1, z2, v]T
## z1, z2 describe position of pod, v describes sideways motion

max_ang_rotation = 0.1*np.pi

class controller_A:
	ku = 0.025
	def __init__(self):
		self.x_prev = 0
		self.y_prev = 0
		self.r = 0

	def getAngle(self, x, y, target_x, target_y):
		# Get the angle [0,2*pi] of the vector going from pod's position to a target
		d = np.array([target_x - x, target_y - y])
		norm_d = np.linalg.norm(d)

		angle = math.acos(d[0] / (norm_d + 1e-16))

		if d[1] < 0:
			angle = 2*np.pi - angle

		return angle

	def rotate(self, vx, vy, phi):
		u =  math.cos(phi)*vx + math.sin(phi)*vy
		v = -math.sin(phi)*vx + math.cos(phi)*vy
		return u, v

	# def get_heading(self, x, y, r_control):
	# 	r_control = np.clip(r_control, -max_ang_rotation, max_ang_rotation)
	#
	# 	return x_heading, y_heading

	# idea of cascading control: calculate reference to track
	def calculate(self, x, y, target_x, target_y):
		# calculate velocities in x and y
		vx = x - self.x_prev
		vy = y - self.y_prev
		self.x_prev = x
		self.y_prev = y

		phi = self.getAngle(x,y, target_x, target_y)
		u, v = self.rotate(vx,vy, phi)

		# calculate states
		z1 =  math.cos(phi)*(x - target_x) + math.sin(phi)*(y - target_y)
		z2 = -math.sin(phi)*(x - target_x) + math.sin(phi)*(y - target_y)

		# choose inputs u, r:
		u_control = -z1+ math.sqrt(pow(v, 2)/4 + pow(z2,2)/4)
		r_control = (4*v +2*z2)/math.sqrt(pow(v,2) + pow(z2,2))


		# apply control law (5) to get thrust (tau_u)
		thrust = self.ku * u_control
		print(thrust)
		# apply control law (6) to get desired angular accelleration (tau_r)
		# tau_r = -u_control*r_control - kr*(r_control - u_control)

		#convert desired angular accelleration to target heading
		#self.get_heading(x,y, r_control)
		x_heading = target_x
		y_heading = target_y

		return thrust, x_heading, y_heading

		
