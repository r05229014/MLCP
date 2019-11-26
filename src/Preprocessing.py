import numpy as np
import sys
import random
import pickle
import os 
from sklearn.preprocessing import StandardScaler


def load_data(path, res, TEST_SPLIT):
	print('Loading data from %s' %path+res)
	TEST_SPLIT = TEST_SPLIT
	
	with open(path + res + '.pkl', 'rb') as f:
		case = pickle.load(f)
	for key, value in case.items():
		case[key] = np.swapaxes(value, 1,3)
		#print(case[key].shape)
	X = np.concatenate((case['u'], case['v'], case['th'], case['qv'], case['cape']), axis=-1)
	y = case['sigma']
	
	# shuffle
	indices = np.arange(X.shape[0])
	nb_test_samples = int(TEST_SPLIT * X.shape[0])
	np.random.shuffle(indices)
	X = X[indices]
	y = y[indices]

	X_train = X[nb_test_samples:]
	X_test = X[0:nb_test_samples]
	y_train = y[nb_test_samples:]
	y_test = y[0:nb_test_samples]

	print('X_train shape is : ', X_train.shape)
	print('X_test shape is : ', X_test.shape)
	print('y_train shape is : ', y_train.shape)
	print('y_test shape is : ', y_test.shape)
	print('\n')

	return X_train, X_test, y_train, y_test


def Preprocessing_Linear(X_train, X_test, y_train, y_test):
	print('Preprocessing for Linear Regression model~~~~')
	X_train = X_train.reshape(-1,4)
	X_test = X_test.reshape(-1,4)
	y_train = y_train.reshape(-1,1)
	y_test = y_test.reshape(-1,1)

	print('X_train shape now is : ', X_train.shape)
	print('X_test shape now is : ', X_test.shape)
	print('y_train shape now is : ', y_train.shape)
	print('y_test shape now is : ', y_test.shape)
	
	return X_train, X_test, y_train, y_test

def Preprocessing_DNN(X_train, X_test, y_train, y_test):
	X_train = X_train.reshape(-1,5)
	X_test = X_test.reshape(-1,5)
	y_train = y_train.reshape(-1,1)
	y_test = y_test.reshape(-1,1)

	sc = StandardScaler()

	# normalize
	print('Normalizing~~~~~~~~~')
	for feature in range(5):
		X_train[:, feature:feature+1] = sc.fit_transform(X_train[:, feature:feature+1])
		X_test[:, feature:feature+1] = sc.fit_transform(X_test[:, feature:feature+1])

	print('X_train shape now is : ', X_train.shape)
	print('X_test shape now is : ', X_test.shape)
	print('y_train shape now is : ', y_train.shape)
	print('y_test shape now is : ', y_test.shape)
	
	return X_train, X_test, y_train, y_test


def main():
	path = '../data/pickle/'
	res = 'd32'
	X_train, X_test, y_train, y_test = load_data(path, res, TEST_SPLIT = 0.2)
	X_train, X_test, y_train, y_test = Preprocessing_DNN(X_train, X_test, y_train, y_test)
	print(y_test[y_test>0])
main()
