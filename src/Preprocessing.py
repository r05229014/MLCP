import numpy as np
import sys
import random
import pickle
import os 
from sklearn.preprocessing import StandardScaler


def PBCs(array, size):
	# size is your arr's size  3*3(3) 5*5(5) 7*7(7) 9*9(9) etc
	out = np.pad(array, ((0,0), (0,0), (size,size), (size,size), (0,0)), 'wrap')
	return out


def CNN3D_type_x(arr,size):
    # size is your arr's size 3*3(size=3), 5*5(size=5)...etc
    out = np.zeros((arr.shape[0]*(arr.shape[2]-(size-1))*(arr.shape[3]-(size-1)), arr.shape[1], size, size, arr.shape[4]), dtype='float16')

    count = 0
    for t in range(arr.shape[0]):
        for x in range(0, arr.shape[2]-(size-1)):
            for y in range(0, arr.shape[3]-(size-1)):
                out[count] = arr[t, :, x:x+size,  y:y+size, :]
                
                count  +=1  
    print("X shape : ", out.shape)
    return out


def CNN3D_type_y(arr):
	out = np.zeros((arr.shape[0]*arr.shape[2]*arr.shape[3], arr.shape[1], 1, 1, arr.shape[4]), dtype='float16')

	count = 0
	for t in range(arr.shape[0]):
		for x in range(0 ,arr.shape[2]):
			for y in range(0, arr.shape[3]):
				out[count] = arr[t, :, x:x+1, y:y+1, :]  

				count += 1 
	#out = np.squeeze(out)
	#out = out.reshape(out.shape[0], 5, 1) # LRCN
	#out = out.reshape(out.shape[0], 5) # 3D CNN
	out = out.reshape(out.shape[0], 1)

	print('y shape : ', out.shape)
	return out


def load_data(path, res, TEST_SPLIT, neighbor):
	PCBs_dict = {3:1, 5:2, 7:3}
	pkl = os.path.join(path, res)
	TEST_SPLIT = TEST_SPLIT
	print(f'Loading data from {pkl} with {TEST_SPLIT} validation data \n')

	# load pickle and span the dimension of ['mcape', 'vimfc', 'sigma']
	with open(pkl, 'rb') as f:
		case = pickle.load(f)
	
	for key, value in case.items():
		if key in ['mcape', 'vimfc']:
			case[key] = np.repeat(value[:, np.newaxis, :, :], 34, axis=1)
			
	# span dimension and concatenate
	for key, value in case.items():
		mean = value.mean()
		std = value.std()
		case[key] = (case[key] - mean)/ std
		case[key] = case[key][..., np.newaxis]
	X = np.concatenate((case['u'], case['v'], case['t'], case['q'], case['h'], case['mcape'], case['vimfc']), axis=-1)
	y = case['sigma'][:, np.newaxis, :, :, :]
	
	# PBCs
	X = PBCs(X, PCBs_dict[neighbor])
	
	# CNN3D input and output
	X = CNN3D_type_x(X, neighbor)
	y = CNN3D_type_y(y)
	sample = sorted(random.sample(list(np.arange(X.shape[0])), k=10752))
	X, y = X[sample], y[sample]
	print(sample)
	print(X.shape,'!!!!!')
	print(y.shape, '!!!!')

	# shuffle and split data
	indices = np.arange(X.shape[0])
	nb_test_samples = int(TEST_SPLIT * X.shape[0])
	np.random.shuffle(indices)
	X = X[indices]
	y = y[indices]

	X_train = X[nb_test_samples:]
	X_test = X[0:nb_test_samples]
	y_train = y[nb_test_samples:]
	y_test = y[0:nb_test_samples]

	print(f'X_train shape is : {X_train.shape}')
	print(f'X_test shape is : {X_test.shape}')
	print(f'y_train shape is : {y_train.shape}')
	print(f'y_test shape is : {y_test.shape}')
	return X_train, X_test, y_train, y_test

# if __name__ == '__main__':
# 	root = '/home/erica/MLCP/data/pickles/'
# 	pkls = os.listdir(root)
# 	res = pkls[0]
# 	neighbors = 3

# 	X_train, X_test, y_train, y_test = load_data(root, res, 0.2, neighbors)