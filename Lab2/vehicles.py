
import matplotlib
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



if __name__ == "__main__":
	vehic = pd.read_csv('./vehicles.csv')
	sns_plot = sns.lmplot(vehic.columns[0], vehic.columns[1], data=vehic, fit_reg=False)

	sns_plot.axes[0,0].set_ylim(0,)
	sns_plot.axes[0,0].set_xlim(0,)

	sns_plot.savefig("scaterplot_vehic.png",bbox_inches='tight')