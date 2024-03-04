import re
import os
import pandas as pd

def read_cosmologies_info():
	file = os.path.join('data', 'cosmologies.csv')

	df = pd.read_csv(file, delimiter='\t', skiprows=1, names=['id', 'Omega_m', 'S_8', 'h', 'w_0', 'sigma_8', 'Omega_cdm'])

	return df

cosmologies_info = read_cosmologies_info()

def get_cosmological_parameters(cosmology_id):
	if str(cosmology_id) in cosmologies_info['id'].values:
		return cosmologies_info[cosmologies_info['id'] == str(cosmology_id)]
	else:
		patt = re.compile('.*Cosmol([0-9]+|fid)')

		if mat := patt.match(cosmology_id):
			return cosmologies_info[cosmologies_info['id'] == mat.group(1)]
		else:
			print(f'Assuming {cosmology_id} is fiducial cosmology')
			return cosmologies_info[cosmologies_info['id'] == 'fid']