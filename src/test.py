import numpy as np
from netCDF4 import Dataset
import os 
import pickle 
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Flatten, Dropout
from keras.callbacks import *
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.utils import multi_gpu_model
from sklearn.linear_model import LinearRegression

#from Preprocessing import load_alldata, Preprocessing_DNN
#from config import ModelMGPU

def load_data(path, ):

  sc = StandardScaler()

  u = Dataset(path + 'd32_u.nc')
  u = u.variables['u'][:]
  u = np.swapaxes(u, 1,3)
  u = u.reshape(-1,1)
  u = sc.fit_transform(u)
  u = u.reshape(690, 16,16,1)

  v = Dataset(path + 'd32_v.nc')
  v = v.variables['v'][:]
  v = np.swapaxes(v, 1,3)
  v = v.reshape(-1,1)
  v = sc.fit_transform(v)
  v = v.reshape(690, 16,16,1)
 
  th = Dataset(path + 'd32_th.nc')
  th = th.variables['th'][:]
  th = np.swapaxes(th, 1,3)
  th = th.reshape(-1,1)
  th = sc.fit_transform(th)
  th = th.reshape(690, 16,16,1)
 
  qv = Dataset(path + 'd32_qv.nc')
  qv = qv.variables['qv'][:]
  qv = np.swapaxes(qv, 1,3)
  qv = qv.reshape(-1,1)
  qv = sc.fit_transform(qv)
  qv = qv.reshape(690, 16,16,1)
 
  sigma = Dataset(path + 'd32_sigma.nc')
  sigma = sigma.variables['sigma'][:]
  sigma = np.swapaxes(sigma, 1,3)

  X = np.concatenate((u,v,th,qv), axis=-1)
  X = X.reshape(-1,4)
  sigma = sigma.reshape(-1,1)
  print(X.shape)
  print(sigma.shape, '!!!!')

  return X, sigma

def preprocessing(X,y, TEST_SPLIT=0.2):
  indices = np.arange(X.shape[0])
  nb_test_samples = int(TEST_SPLIT * X.shape[0])
  np.random.shuffle(indices)
  X_train = X[nb_test_samples:]
  X_test = X[0:nb_test_samples]
  y_train = y[nb_test_samples:]
  y_test = y[0:nb_test_samples]
  
  return X_train, X_test, y_train, y_test


def DNN():
  print("Build model!!")
  model = Sequential()
  model.add(Dense(500, activation = 'selu', input_shape=(4,)))
#  model.add(Dense(512, activation = 'relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
#  model.add(Dense(512, activation = 'relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
#  model.add(Dense(512, activation = 'relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
#  model.add(Dense(512, activation = 'relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
  model.add(Dense(1, activation='relu'))
  return model


path = '../d32/'
X,y = load_data(path)
X_train, X_test, y_train, y_test = preprocessing(X,y)
#linear_model = LinearRegression()
#linear_model.fit(X_train, y_train)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
model = DNN()
#print(model.summary())
model.compile(optimizer='adam', loss='mean_squared_error')

dirpath = "../model/Linear_d32/"
if not os.path.exists(dirpath):
  os.mkdir(dirpath)


filepath= dirpath + "/weights-improvement-{epoch:03d}-{loss:.3e}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
	                                save_best_only=False, period=2)
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# training
history = model.fit(X_train , y_train, validation_data = (X_test, y_test), batch_size=512, epochs=10, shuffle=True, callbacks = [checkpoint])


y_pre = model.predict(X_test)
import matplotlib.pyplot as plt

plt.scatter(y_pre, y_test)
plt.savefig('./d32_DNN.png')
plt.show()
# save history<
history_path = '../history/DNN_10layer_512/'
if not os.path.exists(history_path):
  os.mkdir(history_path)
with open(history_path + 'DNN.pkl', 'wb') as f:
  pickle.dump(history.history, f)

                              

