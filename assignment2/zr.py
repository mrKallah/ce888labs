from train import *

def main():
	print("reading data")
	x_train, y_train, x_test, y_test, headers, feature_names = get_train_test("train.csv", "test.csv")
	print("creating model")
	
	import zero_rule as zr
	model = zr.ZeroRule()

	print("training")
	model.fit(x_train, y_train)

	print("predicting")
	y_pred = model.predict(x_test)

	eval(y_test, y_pred)

# returns 
# Accuracy = 0.1111111111111111
# f1 score = 0.02222222222222222
# [[10000     0     0     0     0     0     0     0     0]
 # [10000     0     0     0     0     0     0     0     0]
 # [10000     0     0     0     0     0     0     0     0]
 # [10000     0     0     0     0     0     0     0     0]
 # [10000     0     0     0     0     0     0     0     0]
 # [10000     0     0     0     0     0     0     0     0]
 # [10000     0     0     0     0     0     0     0     0]
 # [10000     0     0     0     0     0     0     0     0]
 # [10000     0     0     0     0     0     0     0     0]]

if __name__ == "__main__":
	main()