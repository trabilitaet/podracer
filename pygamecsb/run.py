import numpy as np
import os
n_sims = 100

for i in range(n_sims):
	seed = np.random.randint(0, 9999)
	print('seed', seed)
	os.system('python csb.py ' + str(seed) + ' ' + str(0))
	os.system('python csb.py ' + str(seed) + ' ' + str(1))
	print('done')