import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

scores_NMPC = np.genfromtxt('scoresNMPC')
scores_PID = np.genfromtxt('scoresPID')

diff = scores_NMPC - scores_PID
n_samples = scores_PID.size

win_MPC = 0
win_PID = 0
for i in range(n_samples):
	if diff[i] < 0:
		win_MPC +=1
	if diff[i] > 0:
		win_PID +=1
win_percent_MPC = win_MPC*100/n_samples
win_percent_PID = win_PID*100/n_samples
print('Percentage won by MPC: ', win_percent_MPC)
print('Percentage won by PID: ', win_percent_PID)

diff_norm = np.zeros(n_samples)
for i in range(scores_PID.size):
	diff_norm[i] = (scores_NMPC[i]-scores_PID[i])/scores_PID[i]

bin_edges = [-0.55,-0.50,-0.45,-0.40,-0.35,-0.30,-0.25,-0.20,-0.15,-0.10,-0.05,0.0,0.05,0.10,0.15,0.20,0.25]
bin_centers = [x+0.075 for x in bin_edges[:-1]]
bin_indices = np.digitize(diff_norm, bin_edges)

n_bins = len(bin_edges)-1
counts = np.zeros(n_bins)
for index in range(n_samples):
	bin_index = bin_indices[index]-1
	counts[bin_index] += 1

err = np.sqrt(counts)/n_samples

plt.bar(bin_centers, counts/n_samples,facecolor='#0097AC', alpha=0.8, width=0.05, color='k', yerr=err,zorder=2)

# plt.hist(diff_norm, facecolor='#0097AC', alpha=0.8, bins = bin_edges, weights = np.ones(n_samples)/n_samples,zorder=2)

plt.xlabel('Normalized lap time difference')
plt.ylabel('Probability')
plt.axvspan(0, 0.30, alpha=0.5, color='#FF8642',lw=0,zorder=1)
plt.axvspan(-0.55, 0, alpha=0.5, color='#668D3C',lw=0,zorder=1)
plt.xlim(-0.55,0.30)

plt.annotate('PID wins', xy=(0.25, 0.2),
			xytext=(0.15, 0.2),
            arrowprops=dict(facecolor='black'),
            horizontalalignment='right', verticalalignment='center',
            zorder=2,
            )
plt.annotate('MPC wins', xy=(-0.52, 0.2),
			xytext=(-0.27, 0.2),
            arrowprops=dict(facecolor='black'),
            horizontalalignment='right', verticalalignment='center',
            zorder=2,
            )
vals = plt.gca().get_xticks()
plt.gca().set_xticklabels(['{:,.2%}'.format(x) for x in vals])
# plt.gca().set_xticklabels(['{:,.2f}'.format(x*100) for x in vals])
plt.savefig('hist.pdf', format = 'pdf')
# plt.show()