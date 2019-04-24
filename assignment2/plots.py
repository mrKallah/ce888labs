import numpy as np
import matplotlib.pyplot as plt
from load_test import *
from train import *

def plot_confusion_matrix(cm):
	cm = np.asarray(cm)
	num_classes = cm[0]
	plt.matshow(cm)
	plt.colorbar()
	plt.axis('off')
	plt.show()
	
def tree2pdf(model):
	import graphviz 
	from sklearn import tree
	dot_data = tree.export_graphviz(model, out_file=None) 
	graph = graphviz.Source(dot_data) 
	dot_data = tree.export_graphviz(model, out_file=None, 
                      filled=True, rounded=True,  
                      special_characters=True) 
	
	graph = graphviz.Source(dot_data) 
	graph.render("model")


def plot_distribution(values, name):
	unique, counts = np.unique(values, return_counts=True)
	plt.bar(unique, counts, 1)
	print("{} class distribution = {}".format(name, counts))
	plt.title('{} Frequency'.format(values))
	plt.xlabel(values)
	plt.ylabel('Frequency')
	plt.show()


def main():
	import numpy as np
	print("reading data")
	train_header, train = read_csv("train.csv")
	train = np.asarray(train[:,2])
	
	x_train, y_train, x_test, y_test, headers, feature_names = get_train_test("train.csv", "test.csv")
	feature_names, model = load_files()
	cm = [[9658, 1, 15, 0, 313, 5, 3, 5, 0], [713, 7747, 920, 0, 180, 79, 336, 24, 1], [1222, 68, 6855, 4, 170, 72, 1492, 89, 28], [677, 16, 740, 6692, 140, 1001, 674, 60, 0], [3897, 8, 24, 58, 6012, 0, 1, 0, 0], [211, 652, 84, 160, 51, 8619, 151, 38, 34], [1200, 67, 952, 33, 190, 115, 7376, 66, 1], [203, 1128, 157, 580, 92, 145, 103, 6955, 637], [448, 21, 21, 32, 115, 610, 74, 13, 8666]]
	print("done")
	
	plot_distribution(y_train, "class")
	# plot_distribution(train, "feature") # <- slow
	plot_confusion_matrix(cm)
	# tree2pdf(model)



if __name__ == "__main__":
	main()