import os

import matplotlib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def _generate_column_names():
	col_names = ['RA', 'DEC', 'eps_data1', 'eps_data2', 'w', 'z', 'mbias_arun', 'mbias_angus'] + \
				[elem.format(n=n) for n in range(1,11) for elem in ['gamma1_cone{n}', 'gamma2_cone{n}', 'kappa_cone{n}']]

	return col_names


def read_data_file(folder, file_name):
	col_names = _generate_column_names()

	data = pd.read_csv(
		os.path.join(folder, file_name),
		delimiter='\s',
		names=col_names,
		header=None,
	)

	print('RA=', np.min(data.RA), ',', np.max(data.RA))
	print('DEC=', np.min(data.DEC), ',', np.max(data.DEC))

	print(np.sqrt(data.shape[0]))

	create_skymap(data, cone_number=1)

	peak_detection(data)


def peak_detection(data):
	from lenspack.utils import bin2d
	from lenspack.image.inversion import ks93
	from lenspack.peaks import find_peaks2d

	# Bin ellipticity components based on galaxy position into a 128 x 128 map
	e1map, e2map = bin2d(data['RA'], data['DEC'], v=(data['eps_data1'], data['eps_data2']), npix=32)
	# npix refers to the smoothing scale, lower npix, larger smoothing scale

	# Recover convergence via Kaiser-Squires inversion
	kappaE, kappaB = ks93(e1map, e2map)

	# Detect peaks on the convergence E-mode map
	x, y, h = find_peaks2d(kappaE, threshold=0.03, include_border=True)

	# Plot peak positions over the convergence
	fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
	mappable = ax.imshow(kappaE, origin='lower', cmap='bone')
	ax.scatter(y, x, s=10, c='orange')  # reverse x and y due to array indexing
	ax.set_axis_off()
	fig.colorbar(mappable)


def create_skymap(data, cone_number=1):
	fig, ax = plt.subplots()

	cax = ax.scatter(data['RA'], data['DEC'], s=3, c=data[f'kappa_cone{cone_number}'])

	fig.colorbar(cax)


def create_gamma_kappa_hists(data):
	for i in range(1, 11):
		# Create basic statistics plots for each cone
		# For funsies

		fig, axes = plt.subplots(nrows=3, sharex=True)

		for j, col in enumerate([f'gamma1_cone{i}', f'gamma2_cone{i}', f'kappa_cone{i}']):
			axes[j].hist(data[col])
			axes[j].set_ylabel(col)


if __name__ == '__main__':
	read_data_file('data', 'KiDS1000_MocksCat_SLICS_HR_5_LOSALL_R1.dat')

	plt.show()