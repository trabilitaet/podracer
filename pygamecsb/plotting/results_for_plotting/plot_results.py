import numpy as np
import math
from matplotlib import pyplot as plt
from scipy import stats

names = ['88200','81669','82055', '30753']

# names = ['10131','30753','3472','38549','45998','50397','51447','7414','80949','89529']

gamewidth = 16000
gameheight = 9000

number = 0
for name in names:
	number+= 1
	PID = np.load('data/PID_' + name + '.npy').reshape(-1,4)
	MPC = np.load('data/NMPC_' + name + '.npy').reshape(-1,4)
	checkpoints = np.load('data/checkpoints_' + name + '.npy')
	score_PID = PID.shape[0]
	score_MPC = MPC.shape[0]
	diff = score_MPC - score_PID
	print(diff)
	winner = 'PID' if diff >= 0 else 'MPC'

	Px = PID[:-1,0]
	Py = PID[:-1,1]
	Pa = PID[:-1,2]*5
	Pthet = PID[:-1,3]
	Mx = MPC[:-1,0]
	My = MPC[:-1,1]
	Ma = MPC[:-1,2]*5
	Mthet = MPC[:,3]

	# COLORS:
	# 668D3C green
	# 0097AC blue
	# C0362C red
	# FF8642 orange

	####################################################################
	# PLOTTING
	####################################################################
	type = 'both'
	plt.clf()
	plt.xlabel('x-coordinate')
	plt.ylabel('y-coordinate')

	plt.plot(Px,Py, color = '#0097AC', label='PID', linestyle = '-',zorder=1, alpha=0.8)
	plt.plot(Mx,My, color = '#C0362C', label='MPC', linestyle = '-',zorder=1, alpha=0.8)
	for i in range(Px.shape[0]):
		x,y,a,theta = Px[i],Py[i],Pa[i],Pthet[i]
		plt.arrow(x,y,a*math.cos(theta),a*math.sin(theta), width = 0.5, head_width = 80, head_length=100, color = '#668D3C',zorder=2)
	for i in range(Mx.shape[0]):
		x,y,a,theta = Mx[i],My[i],Ma[i],Mthet[i]
		plt.arrow(x,y,a*math.cos(theta),a*math.sin(theta), width = 0.5, head_width = 80,head_length=100, color = '#FF8642',zorder=2)


	#checkpoints
	plt.plot(checkpoints[:1,0],checkpoints[:1,1], color='#C0362C', marker='s',zorder=1)
	for i in range(1,checkpoints.shape[0]-1):
		# plt.plot(checkpoints[i,0],checkpoints[i,1], color='#816C5B', marker='.',zorder=1)
		circle = plt.Circle((checkpoints[i,0],checkpoints[i,1]),450,
			facecolor='#816C5B', alpha = 0.3, fill=True,linewidth=0.5)
		plt.gca().add_artist(circle)

	plt.legend(title='Trajectory of controller:')
	plt.title('Trajectory ' + str(number) + ', Winner: ' + winner + ' by ' + str(diff) + ' ticks')


	#matplotlib
	plt.grid()
	ax = plt.gca()
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.set_aspect('equal', adjustable='box')
	plt.xlim(0,gamewidth)
	plt.xticks(1000*np.arange(17))
	plt.ylim(0,gameheight)
	ax.tick_params(axis='both', which='major', labelsize=6)
	ax.tick_params(axis='both', which='minor', labelsize=4)
	plt.savefig('figures/fig_' + str(number) + '.pdf', format = 'pdf')

