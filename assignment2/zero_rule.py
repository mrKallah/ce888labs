import numpy as np
class ZeroRule():
	def __init__(self):
		self.best = None
	def fit(self, x, y):
		# x is not used, but kept to keep to same format as sklearn
	
		#conform to np array
		x = np.asarray(x)
		y = np.asarray(y)
		
		
		#get unique values from y
		unique = np.unique(y, axis=0)
		occurances = []
		
		# check occurances of unique values in y
		for uni in unique:
			occur = 0
			for _y in y:
				if np.array_equal([uni], [_y]):
					occur += 1
			occurances.append(occur)
		
		# find the index of best result
		best = 0
		for i in range(len(occurances)):
			if occurances[best] < occurances[i]:
				best = i
		
		# set the most occuring class as best
		self.best = unique[best]

	def predict(self, x):
		y_pred = []
		for i in x:
			y_pred.append(self.best)
		return np.asarray(y_pred)