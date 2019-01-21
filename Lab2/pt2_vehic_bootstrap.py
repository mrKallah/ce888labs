import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import numpy as np
import random
from bootstrap import boostrap


def run(data, iter):
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

	sns_plot.savefig("bootstrap_confidence{}.png".format(iter), bbox_inches='tight')
	sns_plot.savefig("bootstrap_confidence{}.pdf".format(iter), bbox_inches='tight')


	print ("Mean: {}".format(np.mean(data)))
	print ("Var: {}".format(np.var(data)))


if __name__ == "__main__":
	df = pd.read_csv('./vehicles.csv')

	x = df.values.T[0]
	y = df.values.T[1]
	
	x = x[~pd.isnull(x)]
	y = y[~pd.isnull(y)]
	
	run(x, 0)
	
	run(y, 1)



	