import sys, argparse, os, time, pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from cnnpreprocessing import *
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def CNN(task,size):
	print("Build model!!")
	model = Sequential()
	model.add(Convolution2D(32, (2, 2), use_bias=True, padding='SAME',strides=1, activation='elu', input_shape=(int(size),int(size),int(task))))
	model.add(MaxPooling2D(pool_size=(3, 3)))
	model.add(Flatten())
	model.add(Dense(256, activation='elu'))
	model.add(Dense(256, activation='elu'))
	model.add(Dense(1, activation='elu'))

	# optimize 
	adam = Adam()

	print('Compiling model...')
	model.compile(loss='mse', optimizer=adam)
	print(model.summary())
	
	return model


def main():
	act = sys.argv[1]
	res = sys.argv[2]
	task = sys.argv[3]
	size = sys.argv[4]
	path = '../data/pickle/'
	model_save_path = '../model/CNN_'+res+'/'
	X,y = load_data(path, res)

	X,y = X[0:666], y[0:666]

	print('\n\nActual use: X and y',X.shape, y.shape)
	X = pool_reflect(X,int(size))
	X = cnn_type_x(X,int(size))
	y = cnn_type_y(y)
	X_train, X_test, y_train, y_test = split_shuffle(X,y)

	if act == 'train':

		model = CNN(task, size)
		if not os.path.exists(model_save_path):
			os.mkdir(model_save_path)
		filepath = model_save_path + "/"+ task +"_feature-1101-{epoch:03d}-{loss:.3e}-size_"+size+".hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='val_loss',save_best_only=True, period=1)
		earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
		history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=512,epochs=500, shuffle=True, callbacks=[checkpoint, earlystopper])
		with open('../history/CNN/'+ task +'_feature_1101_' + res + 'size_'+size+'.pkl' ,'wb') as f:
			pickle.dump(history.history, f)
		cost = model.evaluate(X_test, y_test, batch_size=1024)
		print(cost)

	elif act == 'test':
		load_path = sys.argv[5]
		model = load_model(load_path)
		cost = model.evaluate(X_test, y_test, batch_size=1024)
		print('RMSE = ',cost)
		y_pre = model.predict(X_test, batch_size=1024)
		
		
		plt.figure(figsize=(4,4))
		plt.title('CNN pre '+task +' feature in '+res +'with size :'+size)
		plt.scatter(y_pre, y_test)
		plt.xlim(0,1)
		plt.ylim(0,1)
		plt.xlabel('predict')
		plt.ylabel('True')
		plt.grid(True)
		plt.text(0.4, 0.8, 'RMSE = %.9f'%cost)
		plt.savefig('../img/CNN/CNN_'+res+'_'+task+'feature_size'+size+'.png', dpi=300)

if __name__ == '__main__':
	main()
