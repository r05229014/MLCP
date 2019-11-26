import sys, argparse, os, time, pickle
import numpy as np

from sklearn.preprocessing import StandardScaler


def load_data(path, res):
	print('Loading data from %s' %path+res)
	print('Also normalize data!')
	
	sc = StandardScaler()
	with open(path + res + '.pkl', 'rb') as f:
		case = pickle.load(f)
	y = np.squeeze(case['sigma'])
	y = y.reshape(case['sigma'].shape[0],case['sigma'].shape[-1],case['sigma'].shape[-1],1)
	
	for key, value in case.items():
		case[key] = np.squeeze(value)
		case[key] = case[key].reshape(-1, 1)
		case[key] = sc.fit_transform(case[key])
		case[key] = case[key].reshape(value.shape[0], value.shape[2], value.shape[3], value.shape[1])
		print(key, case[key].shape)
			
	#X = np.concatenate((case['u'], case['v'], case['th'], case['qv']), axis=-1)
	#X = np.concatenate((case['cape'], case['mfc']), axis=-1)
	X = np.concatenate((case['u'], case['v'], case['th'], case['qv'],case['cape'], case['mfc']), axis=-1)
	
	return X,y


def pool_reflect(array, size):
	# size is your arr's size  3*3(3) 5*5(5) 7*7(7) 9*9(9) etc
	new = np.zeros((array.shape[0], array.shape[1]+(size-1), array.shape[2]+(size-1), array.shape[3]))
	for sample in range(array.shape[0]):
		for feature in range(array.shape[3]):
			#print("sample %s" %sample)
			tmp = array[sample,:,:,feature]
			tmp_ = np.pad(tmp, int((size-1)/2), 'wrap')
			new[sample,:,:,feature] = tmp_
	return new


def cnn_type_x(arr, size):
	# size is your arr's size  3*3(3) 5*5(5) 7*7(7) 9*9(9) etc
	out = np.zeros((arr.shape[0]*(arr.shape[1]-(size-1))*(arr.shape[2]-(size-1)), size, size, arr.shape[3]))

	count = 0
	for s in range(arr.shape[0]):
		for x in range(0,arr.shape[1]-(size-1)):
			for y in range(0,arr.shape[2]-(size-1)):
				out[count] = arr[s, x:x+size, y:y+size, :]
				
				count += 1
	print('X sahape : ',out.shape)
	return out


def cnn_type_y(arr):
	out = np.zeros((arr.shape[0]*(arr.shape[1])*(arr.shape[2]), 1, 1, arr.shape[3]))

	count = 0
	for s in range(arr.shape[0]):
		for x in range(0,arr.shape[1]):
			for y in range(0,arr.shape[2]):
				out[count] = arr[s, x, y, :]
				
				count += 1
	out = np.squeeze(out)
	out = out.reshape(out.shape[0], 1)
	print('y shape : ',out.shape)
	return out


def split_shuffle(X,y, TEST_SPLIT=0.2):
	# shuffle
	indices = np.arange(X.shape[0])
	nb_test_samples = int(TEST_SPLIT * X.shape[0])
	np.random.shuffle(indices)
	X = X[indices]
	y = y[indices]
	print(X.shape, '!!!!!!!!!!!!!!!!')
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

if __name__ == '__main__':
	path = '../data/pickle/'
	res = 'd16'
	size = 5
	X,y = load_data(path, res)
	X = pool_reflect(X,size)
	X = cnn_type_x(X,size)
	y = cnn_type_y(y)
	X_train, X_test, y_train, y_test = split_shuffle(X,y)
	print(X_train.shape)
