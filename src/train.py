import numpy as np
import os, sys, pickle

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Convolution3D, MaxPooling3D, Flatten 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from Preprocessing import load_data

def CNN(neighbor):
	model = Sequential()
	model.add(Convolution3D(32, (2,2,2), use_bias=True, padding='SAME', strides=1, activation='relu', input_shape=(34, neighbor, neighbor, 7)))
	model.add(Convolution3D(64, (2,2,2), use_bias=True, padding='SAME', strides=1, activation='relu'))
	model.add(MaxPooling3D(pool_size=(3,3,3)))
	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(1, activation='relu'))
	# optimize 
	adam = Adam()
	model.compile(loss='mse', optimizer=adam)
	return model


def train(root, res, neighbor, model_save_path, TEST_SPLIT):
	print(f'Training {neighbor} Neighbor CNN model with Resolution {res}')
	
	X_train, X_test, y_train, y_test = load_data(root, res, TEST_SPLIT, neighbor)

	# define model callbacks
	model_save_path = os.path.join(model_save_path, res.strip('.pkl'), "Neighbor_%s" %neighbor + '.hdf5')
	checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)
	earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
	model = CNN(neighbor)

	# training
	history = model.fit(X_train, y_train, 
						validation_data=(X_test, y_test), 
						batch_size=256, 
						epochs=500, 
						shuffle=True, 
						callbacks=[checkpoint, earlystopper])

	# save training history
	with open('../history/%s/' %res.strip('.pkl') + 'Neighbor_%s' %neighbor + '.pkl' ,'wb') as f:
		pickle.dump(history.history, f)

if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	
	# hyperparameter
	TEST_SPLIT = 0.2
	root = '/home/erica/MLCP/data/pickles/'
	pkls = os.listdir(root)
	pkls = pkls[1::]
	neighbors = [3,5,7]
	model_save_path = '../model/'
	
	for res in pkls:
		sub_model_folder = os.path.join(model_save_path, res.strip('.pkl'))
		sub_history_folder = os.path.join('../history', res.strip('.pkl'))
		if not os.path.exists(sub_model_folder):
			os.mkdir(sub_model_folder)
		if not os.path.exists(sub_history_folder):
			os.mkdir(sub_history_folder)
		
		for neighbor in neighbors:
			train(root=root ,res=res, neighbor=neighbor, model_save_path=model_save_path, TEST_SPLIT=TEST_SPLIT)
