import argparse
import corner
from emcee import EnsembleSampler
import seaborn as sns
import pandas as pd
import scipy
import glob
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas
from joblib import dump, load
from datetime import datetime

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from analysis.data_compression.compressor import Compressor
from analysis.data_compression.fisher_info_maximizer import FisherInfoMaximizer
from analysis.data_compression.index_compressor import IndexCompressor
from analysis.mcmc import MCMC
import trial
from analysis.map import Map
from analysis.persistence_diagram import PersistenceDiagram
from analysis.persistence_diagram import BettiNumbersGridVarianceMap, PixelDistinguishingPowerMap
import analysis.cosmologies as cosmologies
from analysis.emulator import GPREmulator, MLPREmulator, PerFeatureGPREmulator
from analysis.data_compression.chi_squared_maximizer import ChiSquaredMaximizer
from analysis.data_compression.full_grid import FullGrid
from analysis.pipeline import Pipeline

from utils.file_system import check_folder_exists


slics_truths = [0.2905, 0.826 * np.sqrt(0.2905 / .3), 0.6898, -1.0]


def read_maps(filter_region=None, force_recalculate=False, plots_dir='plots', products_dir='products', save_plots=False):
	pipeline = Pipeline(
		filter_region=filter_region, 
		save_plots=save_plots, force_recalculate=force_recalculate, 
		do_remember_maps=False, bng_resolution=100, three_sigma_mask=True, lazy_load=True,
		plots_dir=plots_dir, products_dir=products_dir
	)
	pipeline.find_max_min_values_maps(save_all_values=False, save_maps=False)
	# pipeline.all_values_histogram()

	pipeline.read_maps()
	pipeline.calculate_variance()

	slics_data = pipeline.slics_data
	cosmoslics_datas = pipeline.cosmoslics_datas
	dist_powers = pipeline.dist_powers

	return slics_data, cosmoslics_datas, dist_powers


def save_datas(slics_data, cosmoslics_datas, dist_powers, dir='cosmology_datas'):
	check_folder_exists(dir)
	dump(slics_data, os.path.join(dir, 'slics_data.joblib'))
	dump(cosmoslics_datas, os.path.join(dir, 'cosmoslics_datas.joblib'))
	dump(dist_powers, os.path.join(dir, 'dist_powers.joblib'))


def load_datas(dir):
	slics_data = load(os.path.join(dir, 'slics_data.joblib'))
	cosmoslics_datas = load(os.path.join(dir, 'cosmoslics_datas.joblib'))
	dist_powers = load(os.path.join(dir, 'dist_powers.joblib'))
	return slics_data, cosmoslics_datas, dist_powers


def create_chisq_comp(slics_data, cosmoslics_datas, dist_powers, chisq_increase, minimum_crosscorr_det=.1, plots_dir='plots'):
	print('Compressing data with ChiSquaredMinimizer...')
	chisqmin = ChiSquaredMaximizer(
		cosmoslics_datas, slics_data, dist_powers, max_data_vector_length=100, 
		minimum_feature_count=50, chisq_increase=chisq_increase, minimum_crosscorr_det=minimum_crosscorr_det,
		add_feature_count=True,
		verbose=False
	)

	check_folder_exists(plots_dir)
	chisqmin.plots_dir = plots_dir

	print('Plotting ChiSquaredMinimizer matrices and data vector...')
	chisqmin.plot_fisher_matrix()
	chisqmin.plot_correlation_matrix()
	chisqmin.plot_covariance_matrix()
	chisqmin.plot_data_vectors(include_slics=True)
	# chisqmin.visualize()

	return chisqmin


def create_fishinfo_comp(slics_data, cosmoslics_datas, dist_powers, fishinfo_increase, minimum_crosscorr_det=.1, plots_dir='plots'):
	print('Compressing data with FisherInfoMaximizer...')
	fishinfo = FisherInfoMaximizer(
		cosmoslics_datas, slics_data, data_vector_length=100, 
		minimum_feature_count=50, fisher_info_increase=fishinfo_increase, minimum_crosscorr_det=minimum_crosscorr_det,
		add_feature_count=True,
		verbose=False
	)

	check_folder_exists(plots_dir)
	fishinfo.plots_dir = plots_dir

	print('Plotting FisherInfoMaximizer matrices and data vector...')
	fishinfo.plot_fisher_matrix()
	fishinfo.plot_correlation_matrix()
	fishinfo.plot_covariance_matrix()
	fishinfo.plot_data_vectors(include_slics=True)

	fishinfo.dist_powers = dist_powers
	# fishinfo.visualize()

	return fishinfo


def create_emulator(compressor, save_name_addition=None, plots_dir='plots'):
	print('Creating emulator...')
	chisq_em = GPREmulator(compressor=compressor)
	chisq_em.plots_dir = plots_dir

	chisq_em.validate(make_plot=True)
	chisq_em.fit()
	chisq_em.plot_predictions_over_parameters(save=True)

	# Pickle the Emulator
	dump(chisq_em, f'emulators/{type(compressor).__name__}_GPREmulator{"" if save_name_addition is None else "_" + save_name_addition}.joblib')

	return chisq_em


def create_full_grid_compressor(slics_pds, cosmoslics_pds):
	print('Creating FullGrid Fisher matrix...')
	# To compare 
	full_grid = FullGrid(cosmoslics_pds, slics_pds)
	full_grid.plot_fisher_matrix()
	# full_grid.plot_crosscorr_matrix()

	return full_grid


def run_with_pickle(pickle_path):
	# Pickle the Emulator
	emu = load(pickle_path)

	return emu


def run_mcmc(emulator, data_vector, p0, nwalkers=100, burn_in_steps=100, nsteps=2500, truths=None, llhood='gauss', plots_dir='plots'):
	with np.errstate(invalid='ignore'):
		ndim = len(p0)

		# init_walkers = np.random.rand(nwalkers, ndim)

		mcmc_ = MCMC(emulator, data_vector, emulator.compressor.slics_covariance_matrix)

		init_walkers = mcmc_.get_random_init_walkers(nwalkers)

		ll = mcmc_.gaussian_likelihood if llhood == 'gauss' else mcmc_.sellentin_heavens_likelihood

		print('Creating EnsembleSampler')
		sampler = EnsembleSampler(nwalkers, ndim, ll)

		print('Running burn in')
		state = sampler.run_mcmc(init_walkers, burn_in_steps)
		sampler.reset()

		print('Running MCMC')
		sampler.run_mcmc(state, nsteps, progress=True, progress_kwargs={'miniters': 1000})

		print('Generating corner plot')
		# Make corner plot
		flat_samples = sampler.get_chain(discard=burn_in_steps, thin=15, flat=True)

		fig = corner.corner(
			flat_samples, labels=cosmologies.get_cosmological_parameters('fid').columns[1:], truths=truths
		)

		fig.savefig(os.path.join(plots_dir, 'corner.png'))

		# Plot chains
		fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
		samples = sampler.get_chain()
		labels = ['$\Omega_m$', '$S_8$', '$h$', '$w_0$']
		for i in range(ndim):
			ax = axes[i]
			ax.plot(samples[:, :, i], "k", alpha=0.3)
			ax.set_xlim(0, len(samples))
			ax.set_ylabel(labels[i])
			ax.yaxis.set_label_coords(-0.1, 0.5)

		axes[-1].set_xlabel("step number")

		fig.savefig(os.path.join(plots_dir, 'chains.png'))


def short_test():
	slics_pds, cosmoslics_pds, dist_powers = read_maps(filter_region=1)

	chisq = create_chisq_comp(slics_pds, cosmoslics_pds, dist_powers, .1, .1, plots_dir='plots/chisq')
	fishin = create_fishinfo_comp(slics_pds, cosmoslics_pds, dist_powers, .05, plots_dir='plots/fishinfo')

	create_emulator(chisq, plots_dir='plots/chisq')
	create_emulator(fishin, plots_dir='plots/fishinfo')


def test_hyperparameters():
	res = {
		'type': [],  # chisq or fisherinfo, type of compressor
		'min_det': [],
		'increase': [],  # chisq_increase or fisher_info_increase
		'final_crosscorr_det': [],
		'vector_length': [],
		'indices': [],
	}

	base_plots_dir = 'plots_test_hyperparams'
	check_folder_exists(base_plots_dir)

	print(f'{datetime.now()} Load datas')
	slics_data, cosmoslics_datas, dist_powers = load_datas('cosmology_datas')

	def save_res(comp: Compressor, comp_type, min_det, inc):

		res['type'].append(comp_type)
		res['min_det'].append(min_det)
		res['increase'].append(inc)
		res['final_crosscorr_det'].append(np.linalg.det(comp.slics_crosscorr_matrix))
		res['vector_length'].append(comp.data_vector_length)

		if len(comp.indices > 0):
			# Identify which zbins are used
			save_indices = [[comp.zbins[ind[0]], ind[1], ind[2], ind[3]] for ind in comp.indices]
			res['indices'].append(save_indices)
		else:
			res['indices'].append(comp.indices)

		df = pandas.DataFrame(res)
		df.to_csv(f'{base_plots_dir}/test_run.csv', index=False)

	print(f'{datetime.now()} Begin test')

	fc_plots = f'{base_plots_dir}/feature_count'
	check_folder_exists(fc_plots)
	featurecount_comp = IndexCompressor(cosmoslics_datas, slics_data, indices=[], add_feature_count=True)
	featurecount_comp.plots_dir = fc_plots
	featurecount_comp.plot_fisher_matrix()
	featurecount_comp.plot_correlation_matrix()
	featurecount_comp.plot_covariance_matrix()
	featurecount_comp.plot_data_vectors(include_slics=True)
	featurecount_comp.plot_data_vectors(save=True, include_slics=True, logy=True, true_value=False)
	# create_emulator(featurecount_comp, save_name_addition='feature_count', plots_dir=fc_plots)
	save_res(featurecount_comp, 'feature_count', 0, 0)
	del featurecount_comp

	for min_det in tqdm([.01, .1]): #+ list(np.logspace(-11, -3, 5)):

		c_tqdm = tqdm(total=8, leave=False)

		for chisq_inc in [.01, .1, .2, .5]:
			plots_dir = f'{base_plots_dir}/plots_det{min_det:.1e}_chisq{chisq_inc}'
			check_folder_exists(plots_dir)
			c = create_chisq_comp(slics_data, cosmoslics_datas, dist_powers, chisq_inc, min_det, plots_dir=plots_dir)

			c.plot_data_vectors(save=True, include_slics=True, logy=True, true_value=False)

			save_res(c, 'chisq', min_det, chisq_inc)

			c_tqdm.update()

			# create_emulator(c, save_name_addition=f'det{min_det:.1e}_chisq{chisq_inc}', plots_dir=plots_dir)
		
		for fishinfo_inc in [.005, .02, .05, .1]:
			plots_dir = f'{base_plots_dir}/plots_det{min_det:.1e}_fishinfo{fishinfo_inc}'
			check_folder_exists(plots_dir)
			c = create_fishinfo_comp(slics_data, cosmoslics_datas, dist_powers, fishinfo_inc, min_det, plots_dir=plots_dir)

			c.plot_data_vectors(save=True, include_slics=True, logy=True, true_value=False)

			save_res(c, 'fishinfo', min_det, fishinfo_inc)

			c_tqdm.update()

			# create_emulator(c, save_name_addition=f'det{min_det:.1e}_fishinfo{fishinfo_inc}', plots_dir=plots_dir)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(prog='KiDS analysis pipeline', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	
	# Pipeline arguments
	map_group = parser.add_argument_group(title='Map reading')
	map_group.add_argument('-r', '--recalculate', action='store_true', help='Force Pipeline to recalculate PersistenceDiagrams and everything else')
	map_group.add_argument('--save-plots-pipeline', action='store_true', help='Flag to save plots produced by Pipeline')

	# Skipping Pipeline, reading CosmologyDatas directly from pickles
	map_group.add_argument('-lcd', '--load-cosm-data', action='store_true', help='Load CosmologyDatas from directory')
	map_group.add_argument('--cosm-data-dir', type=str, default='cosmology_datas', help='Directory in which CosmologyDatas are stored')

	# Emulator to load
	emu_group = parser.add_argument_group(title='Emulator settings')
	emu_group.add_argument('-l', '--load-emulator', action='store_true', help='Flag to set to load from pickle or not. Passed flag means load pickle object')
	emu_group.add_argument('-p', '--pickle-path', type=str, help='Path to Emulator pickle object')

	# MCMC settings
	mcmc_group = parser.add_argument_group('MCMC settings')
	mcmc_group.add_argument('--n-walkers', type=int, help='Number of MCMC walkers', default=500)
	mcmc_group.add_argument('--burn-in-steps', type=int, default=2500, help='Number of burn in steps')
	mcmc_group.add_argument('--n-steps', type=int, default=10000, help='Number of MCMC steps (not including burn in)')
	mcmc_group.add_argument('--likelihood', type=str, default='sellentin-heavens', help='Likelihood function to use')

	gen_group = parser.add_argument_group('General settings')
	gen_group.add_argument('--plots-dir', type=str, help='Directory in which plots are saved', default='plots')
	gen_group.add_argument('--products-dir', type=str, help='Directory in which the products are saved', default='products')
	gen_group.add_argument('-t', '--test', action='store_true', help='Run test function')

	args = parser.parse_args()

	if args.test:
		print('Running test')
		test_hyperparameters()
		sys.exit()

	if not args.load_emulator:
		if not args.load_cosm_data:
			print('Reading maps')
			slics_data, cosmoslics_datas, dist_powers = read_maps(
				force_recalculate=args.recalculate, plots_dir=args.plots_dir, products_dir=args.products_dir, save_plots=args.save_plots_pipeline
			)
			print(f'Saving cosmology datas in {args.cosm_data_dir}')
			save_datas(slics_data, cosmoslics_datas, dist_powers, args.cosm_data_dir)
		else:
			print(f'Loading cosmology datas from {args.cosm_data_dir}')
			slics_data, cosmoslics_datas, dist_powers = load_datas(args.cosm_data_dir)
		# comp = create_chisq_comp(slics_data, cosmoslics_datas, dist_powers, chisq_increase=0.1, minimum_crosscorr_det=0.1, plots_dir=args.plots_dir)
		comp = create_fishinfo_comp(slics_data, cosmoslics_datas, dist_powers, fishinfo_increase=0.05, minimum_crosscorr_det=0.1, plots_dir=args.plots_dir)
		emu = create_emulator(comp, plots_dir=args.plots_dir)
	else:
		print('Loading pickle file', args.pickle_path)
		emu = run_with_pickle(args.pickle_path)

	print(f'Running MCMC with nwalkers={args.n_walkers}, burn_in_steps={args.burn_in_steps}, nsteps={args.n_steps}, llhood={args.likelihood}')
	run_mcmc(emu, emu.compressor.avg_slics_data_vector, p0=np.random.rand(4), truths=slics_truths, 
			nwalkers=args.n_walkers, burn_in_steps=args.burn_in_steps, nsteps=args.n_steps, llhood=args.likelihood,
			plots_dir=args.plots_dir)
	