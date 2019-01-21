import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import numpy as np
import random



def boostrap(sample, sample_size, iterations, cl_upper, cl_lower):
	# create the bootstraps
	new_samples = []
	mean = []
	for iteration in range(iterations):
		x = []
		for j in range(sample_size):
			x.append(sample[random.randint(0,len(sample)-1)])
		new_samples.append(x)
		mean.append(new_samples[iteration:])
	
	data_mean = np.mean(new_samples)
	upper = np.percentile(mean, cl_upper)
	lower = np.percentile(mean, cl_lower)
	
	print(iterations)

	return data_mean, lower, upper


if __name__ == "__main__":
	df = pd.read_csv('./salaries.csv')

	data = df.values.T[1]
	boots = []
	for i in range(100, 100000, 1000):
		boot = boostrap(data, data.shape[0], i, 95, 5)
		boots.append([i, boot[0], "mean"])
		boots.append([i, boot[1], "lower"])
		boots.append([i, boot[2], "upper"])

	df_boot = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])
	sns_plot = sns.lmplot(df_boot.columns[0], df_boot.columns[1], data=df_boot, fit_reg=False, hue="Value")

	sns_plot.axes[0, 0].set_ylim(0,)
	sns_plot.axes[0, 0].set_xlim(0, 100000)

	sns_plot.savefig("bootstrap_confidence.png", bbox_inches='tight')
	sns_plot.savefig("bootstrap_confidence.pdf", bbox_inches='tight')


	print ("Mean: {}".format(np.mean(data)))
	print ("Var: {}".format(np.var(data)))
	


	