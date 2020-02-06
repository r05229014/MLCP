import os, pickle, sys
from netCDF4 import Dataset

def nc2pickle(nc_path, res, save_path):
	# load ncfile
	ncfile_u = Dataset(nc_path + '/u' + '_' + res + '.nc')
	ncfile_v = Dataset(nc_path + '/v' + '_' + res + '.nc')
	ncfile_t = Dataset(nc_path + '/t' + '_' + res + '.nc')
	ncfile_q = Dataset(nc_path + '/q' + '_' + res + '.nc')
	ncfile_h = Dataset(nc_path + '/h' + '_' + res + '.nc')
	ncfile_mcape = Dataset(nc_path + '/mcape' + '_' + res + '.nc')
	ncfile_vimfc = Dataset(nc_path + '/vimfc' + '_' + res + '.nc')
	ncfile_sigma = Dataset(nc_path + '/sigma' + '_' + res + '.nc')

	# load variables
	u = ncfile_u.variables['u'][:]
	v = ncfile_v.variables['v'][:]
	t = ncfile_t.variables['t'][:]
	q = ncfile_q.variables['q'][:]
	h = ncfile_h.variables['h'][:]
	mcape = ncfile_mcape.variables['mcape'][:]
	vimfc = ncfile_vimfc.variables['vimfc'][:]
	sigma = ncfile_sigma.variables['sigma'][:]

	variables_dict = {'u':u, 'v':v, 't':t, 'q': q, 'h':h, 'mcape':mcape, 'vimfc':vimfc, 'sigma':sigma}

	with open(save_path, 'wb') as f:
		pickle.dump(variables_dict, f)

if __name__ == '__main__':
	root = '../data/'
	res = '008km'
	resolutions = ['008km', '016km', '032km', '064km', '128km']

	for res in resolutions:
		nc_path = os.path.join(root, 'ncfile')
		save_path = os.path.join(root, 'pickles', res + '.pkl')
		nc2pickle(nc_path, res, save_path)
