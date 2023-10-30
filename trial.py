import os

import matplotlib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import treecorr

import analysis.athena as athena
from analysis.map import Map
from analysis.persistence_diagram import PersistenceDiagram
import glob


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
	import analysis.treecorr_utils as treecorr_utils
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

	fig.savefig('plots/map.png')


def create_gamma_kappa_hists(data):
	for i in range(1, 11):
		# Create basic statistics plots for each cone
		# For funsies

		fig, axes = plt.subplots(nrows=3, sharex=True)

		for j, col in enumerate([f'gamma1_cone{i}', f'gamma2_cone{i}', f'kappa_cone{i}']):
			axes[j].hist(data[col])
			axes[j].set_ylabel(col)


def all_maps():

	from tqdm import tqdm

	print('Determining max and min values in maps...')
	min_val = {
		0: np.inf,
		1: np.inf
	}
	max_val = {
		0: -np.inf,
		1: -np.inf
	}
	for dir in tqdm(glob.glob('maps/*')):
		if os.path.isdir(dir):
			for i, map_path in enumerate(tqdm(glob.glob(f'{dir}/*.npy'), leave=False)):
				map = Map(map_path)
				map.get_persistence()
				pd = PersistenceDiagram([map])

				for dim in [0, 1]:
					curr_min = np.min(pd.dimension_pairs[dim])
					curr_max = np.max(pd.dimension_pairs[dim])
					if curr_min < min_val[dim]:
						min_val[dim] = curr_min
					if curr_max > max_val[dim]:
						max_val[dim] = curr_max
	
	print('min=', min_val)
	print('max=', max_val)

	data_range = {
		dim : [min_val[dim], max_val[dim]] for dim in [0, 1]
	}

	# TODO: compare SLICS variance with cosmoSLICS variance
	# that is, compare los variance within SLICS to variance between different cosmologies

	# SLICS determines the sample variance, will be a list of persistence diagrams for each line of sight
	slics_pds = []
	# cosmoSLICS is different cosmologies, will be a list of persistence diagrams for each cosmology
	cosmoslics_pds = []
	cosmoslics_uniq_pds = []

	slics_maps = []
	cosmoslics_maps = []

	print('Analyzing maps...')
	for dir in tqdm(glob.glob('maps/*')):
		if os.path.isdir(dir):
			cosm = dir.split('_')[-1]

			cosmoslics = 'Cosmo' in cosm

			curr_cosm_maps = []
			for i, map_path in enumerate(tqdm(glob.glob(f'{dir}/*.npy'), leave=False)):
				# if len(slics_pds) > 5 and not cosmoslics:
				# 	continue
				# if len(cosmoslics_uniq_pds) > 5 and cosmoslics:
				# 	continue

				map = Map(map_path)
				map.get_persistence()
				curr_cosm_maps.append(map)

				pd = PersistenceDiagram([map])
				pd.generate_betti_numbers_grids(resolution=100, data_ranges_dim=data_range)

				# SLICS must be saved at LOS level
				if not cosmoslics:
					slics_pds.append(pd)
					slics_maps.append(map)
				else:
					cosmoslics_uniq_pds.append(pd)
					cosmoslics_maps.append(map)

			if len(curr_cosm_maps) > 0:
				pd = PersistenceDiagram(curr_cosm_maps)
				pd.generate_heatmaps(resolution=100, gaussian_kernel_size_in_sigma=3)
				# pd.add_average_lines()
				pd.generate_betti_numbers_grids(resolution=100, data_ranges_dim=data_range)

				pd.plot()

				# cosmoSLICS must be saved at cosmology level
				if cosmoslics:
					cosmoslics_pds.append(pd)

	print('Calculating SLICS/cosmoSLICS variance maps...')
	slics_bngs = {
		dim: [pd.betti_numbers_grids[dim] for pd in slics_pds] for dim in [0, 1]
	}
	cosmoslics_bngs = {
		dim: [pd.betti_numbers_grids[dim] for pd in cosmoslics_pds] for dim in [0, 1]
	}

	from analysis.persistence_diagram import BettiNumbersGridVarianceMap

	dim = 0
	slics_bngvm_0 = BettiNumbersGridVarianceMap(slics_bngs[dim], birth_range=data_range[dim], death_range=data_range[dim], dimension=dim)
	slics_bngvm_0.save_figure(os.path.join('plots', 'slics'), title='SLICS variance, dim=0')
	dim = 1
	slics_bngvm_1 = BettiNumbersGridVarianceMap(slics_bngs[dim], birth_range=data_range[dim], death_range=data_range[dim], dimension=dim)
	slics_bngvm_1.save_figure(os.path.join('plots', 'slics'), title='SLICS variance, dim=1')

	dim = 0
	cosmoslics_bngvm_0 = BettiNumbersGridVarianceMap(cosmoslics_bngs[dim], birth_range=data_range[dim], death_range=data_range[dim], dimension=dim)
	cosmoslics_bngvm_0.save_figure(os.path.join('plots', 'cosmoslics'), title='cosmoSLICS variance, dim=0')
	dim = 1
	cosmoslics_bngvm_1 = BettiNumbersGridVarianceMap(cosmoslics_bngs[dim], birth_range=data_range[dim], death_range=data_range[dim], dimension=dim)
	cosmoslics_bngvm_1.save_figure(os.path.join('plots', 'cosmoslics'), title='cosmoSLICS variance, dim=1')

	fig, ax = plt.subplots()
	ax.set_title('slics / cosmoslics variance, dim=0')
	imax = ax.imshow((slics_bngvm_0.map / cosmoslics_bngvm_0.map)[::-1, :])
	fig.colorbar(imax)
	fig.savefig(os.path.join('plots', 'slics_cosmoslics_variance_0.png'))
	plt.close(fig)

	fig, ax = plt.subplots()
	ax.set_title('slics / cosmoslics variance, dim=1')
	imax = ax.imshow((slics_bngvm_1.map / cosmoslics_bngvm_1.map)[::-1, :])
	fig.colorbar(imax)
	fig.savefig(os.path.join('plots', 'slics_cosmoslics_variance_1.png'))
	plt.close(fig)

	slics_pd = PersistenceDiagram(slics_maps)
	slics_pd.generate_betti_numbers_grids(data_ranges_dim=data_range)
	cosmoslics_pd = PersistenceDiagram(cosmoslics_maps)
	cosmoslics_pd.generate_betti_numbers_grids(data_ranges_dim=data_range)

	cosmoslics_bngs = {
		dim: np.array([pd.betti_numbers_grids[dim].map for pd in cosmoslics_pds]) for dim in [0, 1]
	}

	for dim in [0, 1]:
		print('cosmoSLICS')
		print(cosmoslics_bngs[dim].shape)
		print('SLICS')
		print(slics_pd.betti_numbers_grids[dim])
		print((cosmoslics_bngs[dim] - slics_pd.betti_numbers_grids[dim]).shape)
		print(np.mean(np.square(cosmoslics_bngs[dim] - slics_pd.betti_numbers_grids[dim]), axis=0).shape)
		dist_power = np.mean(np.square(cosmoslics_bngs[dim] - slics_pd.betti_numbers_grids[dim].map), axis=0) / BettiNumbersGridVarianceMap(slics_bngs[dim], birth_range=data_range[dim], death_range=data_range[dim], dimension=dim).map

		fig, ax = plt.subplots()
		ax.set_title('Pixel distinguishing power')
		imax = ax.imshow(dist_power[::-1, :], extent=(*data_range[dim], *data_range[dim]))
		fig.colorbar(imax)

		fig.savefig(os.path.join('plots', f'pixel_distinguishing_power_{dim}.png'))
		plt.close(fig)


def do_map_stuff():
	filename = os.path.join('maps', 'SN0.27_Mosaic.KiDS1000GpAM.LOS74R1.SS3.982.Ekappa.npy')

	map = Map(filename)
	print(map.map.shape)
	map.plot()
	plt.savefig('plots/mapp.png')

	print(map.get_betti_numbers())

	for i, t in enumerate(np.linspace(map.map.min(), map.map.max(), 5)):
		if i == 0:
			prev_t = t
			continue

		print(f't\',t = {prev_t}, {t}')
		print(map.get_persistent_betti_numbers(prev_t, t))

		prev_t = t

	# map.generate_heatmaps(resolution=1000)

	persax = PersistenceDiagram([map]).ax

	# persax.plot(persax.get_ylim(), persax.get_ylim(), color='gray', linestyle='--')
	# persax.imshow(map.heatmaps[0][:,::-1], extent=(*(map.heatmaps[0].birth_range), *(map.heatmaps[0].death_range)))
	# plt.savefig(os.path.join('plots', 'heatmap_proper_scaling.png'))

	# fig, ax = plt.subplots()
	# ax.imshow(map.heatmaps[0][:,::-1])#[::-1], origin='lower')

	# fig, ax = plt.subplots()
	# ax.imshow(map.heatmaps[1][:,::-1])#[::-1], origin='lower')

	# # Hist of values in map
	# fig, ax = plt.subplots()
	# ax.hist(map.map.flatten())


if __name__ == '__main__':
	# data = read_data_file('data', 'KiDS1000_MocksCat_SLICS_HR_5_LOSALL_R1.dat')

	# create_skymap(data)

	# run_athena(data)

	# df_athena = athena.get_output()

	# plot_correlation_function(df_athena)

	# run_treecorr(data)

	all_maps()


	plt.show()