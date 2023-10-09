import numpy as np
import pandas as pd
import subprocess

PATH_TO_ATHENA = 'athena_1.7/bin/athena'

DEFAULT_CONFIG = {
    'GALCAT1': 'athena_run/gal_cat.csv',
    'GALCAT2': '-',
    'WCORR': 1,
    'SFORMAT': 'standard',
    'SCOORD_INPUT': 'deg',
    'SCOORD_OUTPUT': 'deg',
    'THMIN': 0.1,
    'THMAX': 10,
    'NTH': 100,
    'BINTYPE': 'LOG',
    'RADEC': 1,
    'OATH': 0,
    'SERROR': 'none',
    'NRESAMPLE': 5
}

def convert_dataframe_to_athena_format(data, outfile='athena_run/gal_cat.csv', columns=['RA', 'DEC', 'eps_data1', 'eps_data2', 'w']):
	x = data[columns[0]]
	y = data[columns[1]]
	e1 = data[columns[2]]
	e2 = data[columns[3]]
	w = data[columns[4]]

	df = pd.DataFrame({
		'x': x,
		'y': y,
		'e1': e1,
		'e2': e2,
		'w': w
	})

	df.to_csv(outfile, sep=' ', header=False, index=False)


def create_config_file(config=None, file='athena_run/config.cfg'):
    if config is None:
        config = {}

    with open(file, 'w+') as config_file:
        for key, value in DEFAULT_CONFIG.items():
            if key in config:
                value = config[key]

            config_file.write(f'{key} {value}\n')

def run(catalog='athena_run/gal_cat.csv', config='athena_run/config.cfg'):
    subprocess.run([PATH_TO_ATHENA, '-c', config])


OUTPUT_FILES = {
    'shear-shear': 'xi',
    'shear-position': 'wgl',
    'position-position': 'w'
}


def get_output(type='shear-shear'):
    file_name = OUTPUT_FILES[type]

    df = pd.read_csv(
        file_name, delim_whitespace=True, header=None, 
        names=['theta', 'xi_p', 'xi_m', 'xi_x', 'w', 'sqrt_D', 'sqrt_Dcor', 'n_pair'],
        skiprows=1,
        dtype={
            'theta': np.float64, 
            'xi_p': np.float64,
            'xi_m': np.float64,
            'xi_x': np.float64,
            'w': np.float64,
            'sqrt_D': np.float64,
            'sqrt_Dcor': np.float64,
            'n_pair': int
        }
    )

    print(df.dtypes)

    return df
