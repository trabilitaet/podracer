import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

wins = np.zeros((14))
dnf = np.zeros((14))
wins_fin = np.zeros((14))
for i in range(1,14):
      print(i)
      scores_NMPC = np.genfromtxt('results/MPC' + str(i))
      scores_PID = np.genfromtxt('results/PID' + str(i))

      diff = scores_NMPC - scores_PID
      n_samples = scores_PID.size

      # print(diff)

      win_MPC = 0
      dnf_MPC = 0
      for j in range(n_samples):
            if diff[j] <= 0:
                  win_MPC +=1
            elif 1000 <= diff[j]:
                  dnf_MPC +=1

      wins[i] = win_MPC
      dnf[i] = dnf_MPC
      wins_fin[i] = win_MPC*(100/(100-dnf_MPC)) if not dnf_MPC==100 else 'NaN'

# print(wins)
# print(dnf)

plt.plot(dnf, color='#0097AC', marker='.', linestyle='-.', label='Percentage of matches MPC did not finish (DNF)')
plt.plot(wins, color='#668D3C', marker='o', label='Percentage of total matches won by MPC')
plt.plot(wins_fin, color='#FF8642', marker='d', linestyle='--', label='Percentage of completed matches won by MPC')
plt.xlabel('Np')

plt.legend()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylim(0,135)
plt.xlim(1,14)
plt.yticks(10*np.arange(11))
plt.savefig('varNhist.pdf', format = 'pdf')
# plt.show()