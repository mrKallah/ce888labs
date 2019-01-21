import numpy as np
from scipy.stats import ttest_ind

def power(sample1, sample2, reps, size, alpha):
	sum = 0
	for i in range(reps):
		new_sample1 = np.random.choice(sample1, (iterations, sample_size), replace=True)
		new_sample2 = np.random.choice(sample2, (iterations, sample_size), replace=True)
		t, p = ttest_ind(new_sample1, new_sample2)
		if p < (1-alpha):
			sum += 1
	return sum
		