import numpy as np 
import os, sys
from scipy.io import FortranFile


def dat2npy(file_path, save_path, x, y, z=0):
    '''
    file_path :: path of .dat file
    save_path :: path to save .npy file
    x         :: shape of x axis
    y         :: shape of y axis
    z         :: shape of z axis, defalut 0
    '''
    if z == 0:
        feature = np.fromfile(file_path, dtype=np.float32).reshape(-1, x, y)
        assert feature.shape == (672, x, y)
    else:
        feature = np.fromfile(file_path, dtype=np.float32).reshape(-1, x, y, z)
        assert feature.shape == (672, x, y, 34)
    np.save(save_path, feature)
    print(f'Transfering {file_path} to {save_path} with shape = {feature.shape}')
    return feature

if __name__ == '__main__':
    
    load_folder = '../dat/'
    save_folder = '../npy/'
    dats = os.listdir(load_folder)

    # Transfer dict
    z_shape = {'h':34, 'q':34, 't':34, 'u':34, 'v':34, 'mcape':0, 'sigma':0, 'vimfc':0}
    xy_shape = {'008km':64, '016km':32, '032km':16, '064km':8, '128km':4}

    for dat in dats:
        file_name = dat.split('.')[0]
        feature_name, res = file_name.split('_')[0], file_name.split('_')[1]
        load_path = os.path.join(load_folder, dat)
        save_path = os.path.join(save_folder, feature_name + '_' + res + '.npy')

        z = z_shape[feature_name]
        x, y = xy_shape[res], xy_shape[res]

        #print(load_path, feature_name, res, z, x, y)
        dat2npy(load_path, save_path, x, y, z)

