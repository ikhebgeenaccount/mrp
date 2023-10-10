import os

import matplotlib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import treecorr

import src.athena as athena


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

	return data


def run_athena(data):
	athena.convert_dataframe_to_athena_format(data, 'athena_run/gal_cat.csv')
	athena.create_config_file()
	athena.run()


def run_treecorr(data, out_file='treecorr_2pcf.out'):
	import src.treecorr_utils as treecorr_utils
	cat = treecorr_utils.build_treecorr_catalog(data)
	gg = treecorr.GGCorrelation(min_sep=.1, max_sep=100., bin_size=.1, sep_units='arcmin')

	gg.process(cat)
	gg.write(out_file)

	tpcf_df = treecorr_utils.read_treecorr_result(out_file)

	treecorr_utils.plot_correlation_function(tpcf_df)


def plot_correlation_function(df, theta_col='theta', xi_m_col='xi_m', xi_p_col='xi_p', source='athena'):
	fig, ax = plt.subplots()
	ax.errorbar(df[theta_col], df[xi_p_col], label='$\\xi_+$')
	ax.errorbar(df[theta_col], df[xi_m_col], label='$\\xi_-$')
	# ax.hist(df_athena.xi_p, bins=df_athena.theta + (df_athena.theta[1] - df_athena.theta[0]) / 2, label='$\ksi_+$')
	# ax.hist(df_athena.xi_m, bins=df_athena.theta + (df_athena.theta[1] - df_athena.theta[0]) / 2, label='$\ksi_-$')
	ax.legend()

	fig.savefig(os.path.join('plots', f'2pt_correlation_func_{source}.png'))

	# create_skymap(data, cone_number=1)

	# peak_detection(data)


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


def do_map_stuff():
	from src.map import Map

	filename = os.path.join('maps', 'SN0.27_Mosaic.KiDS1000GpAM.LOS74R1.SS3.982.Ekappa.npy')

	map = Map(filename)
	print(map.map.shape)
	map.plot()

	map.get_persistence()
	persax = map.plot_persistence()
	import sys
	sys.exit()

	persax.plot(persax.get_ylim(), persax.get_ylim(), color='gray', linestyle='--')

	print(map.get_betti_numbers())
	# print(map.get_persistent_betti_numbers())

	map.generate_heatmaps(resolution=1000)
	persax.imshow(map.heatmaps[0][:,::-1], extent=(*(map.heatmaps[0].birth_range), *(map.heatmaps[0].death_range)))

	fig, ax = plt.subplots()
	ax.imshow(map.heatmaps[0][:,::-1])#[::-1], origin='lower')

	fig, ax = plt.subplots()
	ax.imshow(map.heatmaps[1][:,::-1])#[::-1], origin='lower')

	# Hist of values in map
	fig, ax = plt.subplots()
	ax.hist(map.map.flatten())


if __name__ == '__main__':
	data = read_data_file('data', 'KiDS1000_MocksCat_SLICS_HR_5_LOSALL_R1.dat')

	create_skymap(data)

	# run_athena(data)

	# df_athena = athena.get_output()

	# plot_correlation_function(df_athena)

	# run_treecorr(data)

	do_map_stuff()


	plt.show()