import numpy as np

def eval(y, pred):
	from sklearn.metrics import confusion_matrix
	cm = confusion_matrix(y, pred)

	from sklearn.metrics import accuracy_score
	acc = accuracy_score(y, pred)

	from sklearn.metrics import f1_score
	f1 = f1_score(y, pred, average='weighted')

	print("Accuracy = {}".format(acc))
	print("f1 score = {}".format(f1))
	print(cm)

def read_csv(path):
	import csv
	file = []
	with open(path) as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			file.append(np.asarray(row))
	return np.asarray(file[:1]), np.asarray(file[1:])

def str2int(arr):
	new_arr = []
	for i in arr:
		row = []
		for j in range(9):
			if(j=="."):
				row.append(0)
			elif(j=="O"):
				row.append(1)
			elif(j=="X"):
				row.append(2)
			else:
				row.append(3)
			
		new_arr.append(row)
	return np.asarray(new_arr)

def get_train_test(train_path, test_path):
	# read files
	train_header, train = read_csv(train_path)
	test_header, test = read_csv(test_path)
	
	assert(np.array_equal(train_header, test_header))
	
	# seperate class from features
	x_train = np.asarray(train[:,2])
	x_test = np.asarray(test[:,2])

	x_train = np.asarray(x_train)
	x_test = np.asarray(x_test)


	all = np.append(x_train, x_test)
	
	# from sklearn.preprocessing import LabelEncoder
	# enc = LabelEncoder()
	
	from sklearn.preprocessing import OneHotEncoder
	enc = OneHotEncoder()
	
	feature_names = np.unique(all)
	
	# all = enc.fit_transform(all).reshape(-1, 1)).toarray()
	all = enc.fit(feature_names.reshape(-1, 1)).transform(all.reshape(-1, 1)).toarray()

	x_train = all[:x_train.shape[0]]
	x_test = all[x_train.shape[0]:]
		
	y_train = np.asarray(train[:,1])
	y_test = np.asarray(test[:,1])

	return x_train, y_train, x_test, y_test, train_header, feature_names
	
def main():
	print("reading data")
	x_train, y_train, x_test, y_test, headers, feature_names = get_train_test("train.csv", "test.csv")
	print("creating model")

	from sklearn import tree
	model = tree.DecisionTreeClassifier()

	# from sklearn.ensemble import RandomForestClassifier
	# model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

	print("training")
	model.fit(x_train, y_train)

	print("predicting")
	y_pred = model.predict(x_test)

	eval(y_test, y_pred)

	import pickle
	filename = 'finalized_model.sav'
	pickle.dump(model, open(filename, 'wb'))
	
	features_dir = 'feature_names.sav'
	pickle.dump(feature_names, open(features_dir, 'wb'))


if __name__ == "__main__":
	main()