
def load_files():
	import pickle
	# load feature names from file
	features_dir = 'feature_names.sav'
	file = open(features_dir,'rb')
	feature_names = pickle.load(file)

	# load model from file
	model_dir = 'finalized_model.sav'
	file = open(model_dir,'rb')
	model = pickle.load(file)
	return feature_names, model

def predict(x, feature_names, model):
	import numpy as np
	from sklearn.preprocessing import OneHotEncoder
	
	# one hot encode input
	enc = OneHotEncoder()
	x_enc = enc.fit(feature_names.reshape(-1, 1)).transform(x.reshape(-1, 1)).toarray()
	
	# predict
	return model.predict(x_enc)

def main():
	import numpy as np
	
	feature_names, model = load_files()
	
	x = np.asarray(["1.....X..."])
	y_true = np.asarray(['7'])
	y_pred = predict(x, feature_names, model)

	print(x)
	print(y_true)
	print(y_pred)
	
if __name__ == "__main__":
	main()