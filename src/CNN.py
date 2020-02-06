import sys, argparse, os, time, pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from cnnpreprocessing import *
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Convolution3D, MaxPooling3D, Flatten 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def CNN(neighbor):
    model = Sequential()
    model.add(Convolution3D(32, (2,2,2), use_bias=True, padding='SAME', strides=1, activation='relu', input_shape=(34, neighbor, neighbor, 7)))
    model.add(Convolution3D(64, (2,2,2), use_bias=True, padding='SAME', strides=1, activation='relu'))
    model.add(Convolution3D(128, (2,2,2), use_bias=True, padding='SAME', strides=1, activation='relu'))
    model.add(MaxPooling3D(pool_size=(2,2,2)))

    model.add(Convolution3D(32, (2,2,2), use_bias=True, padding='SAME', strides=1, activation='relu'))
    model.add(Convolution3D(64, (2,2,2), use_bias=True, padding='SAME', strides=1, activation='relu'))
    model.add(Convolution3D(128, (2,2,2), use_bias=True, padding='SAME', strides=1, activation='relu'))
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    
	model.add(Flatten())
	model.add(Dense(1, activation='relu'))

	model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == '__main__':
	#act = sys.argv[1]
	#res = sys.argv[2]
	#task = sys.argv[3]
	#size = sys.argv[4]
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	
	path = '../data/pickles/'
	res = '128km'
	size = 5
	model_save_path = '../model/CNN.h5

	X,y = load_data(path, res)
	X = pool_reflect(X, size)
	X = cnn_type_x(X, size)
	y = cnn_type_y(y)
	X_train, X_test, y_train, y_test = split_shuffle(X,y)