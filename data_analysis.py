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

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from analysis.data_compression.fisher_info_maximizer import FisherInfoMaximizer
from analysis.mcmc import MCMC
import trial
from analysis.map import Map
from analysis.persistence_diagram import PersistenceDiagram
from analysis.persistence_diagram import BettiNumbersGridVarianceMap, PixelDistinguishingPowerMap
import analysis.cosmologies as cosmologies
from analysis.emulator import GPREmulator, MLPREmulator, PerFeatureGPREmulator
from analysis.data_compression.betti_number_peaks import BettiNumberPeaksCompressor
from analysis.data_compression.chi_squared_minimizer import ChiSquaredMinimizer
from analysis.data_compression.histogram import HistogramCompressor
from analysis.data_compression.number_of_features import NumberOfFeaturesCompressor
from analysis.data_compression.full_grid import FullGrid
from analysis.pipeline import Pipeline


slics_truths = [0.2905, 0.826 * np.sqrt(0.2905 / .3), 0.6898, -1.0]


def run():
	pipeline = Pipeline(save_plots=False, force_recalculate=False, do_remember_maps=False, bng_resolution=100, three_sigma_mask=True)
	pipeline.find_max_min_values_maps(save_all_values=False, save_maps=False)
	# pipeline.all_values_histogram()

	pipeline.read_maps()
	pipeline.calculate_variance()

	slics_pds = pipeline.slics_pds
	cosmoslics_pds = pipeline.cosmoslics_pds
	dist_powers = pipeline.dist_powers

	return slics_pds, cosmoslics_pds, dist_powers


def create_chisq_comp(slics_pds, cosmoslics_pds, dist_powers, chisq_increase, minimum_crosscorr_det=.1, plots_dir='plots'):
	print('Compressing data with ChiSquaredMinimizer...')
	chisqmin = ChiSquaredMinimizer(
		cosmoslics_pds, slics_pds, dist_powers, max_data_vector_length=100, 
		minimum_feature_count=40, chisq_increase=chisq_increase, minimum_crosscorr_det=minimum_crosscorr_det,
		verbose=True
	)

	chisqmin.plots_dir = plots_dir

	print('Plotting ChiSquaredMinimizer matrices and data vector...')
	chisqmin.plot_fisher_matrix()
	chisqmin.plot_correlation_matrix()
	chisqmin.plot_covariance_matrix()
	chisqmin.plot_data_vectors(include_slics=True)
	chisqmin.visualize()

	return chisqmin


def create_fishinfo_comp(slics_pds, cosmoslics_pds, dist_powers, fishinfo_increase, minimum_crosscorr_det=.1, plots_dir='plots'):
	print('Compressing data with ChiSquaredMinimizer...')
	fishinfo = FisherInfoMaximizer(
		cosmoslics_pds, slics_pds, data_vector_length=100, 
		minimum_feature_count=40, fisher_info_increase=fishinfo_increase, minimum_crosscorr_det=minimum_crosscorr_det,
		verbose=True
	)

	fishinfo.plots_dir = plots_dir

	print('Plotting ChiSquaredMinimizer matrices and data vector...')
	fishinfo.plot_fisher_matrix()
	fishinfo.plot_correlation_matrix()
	fishinfo.plot_covariance_matrix()
	fishinfo.plot_data_vectors(include_slics=True)

	fishinfo.dist_powers = dist_powers
	fishinfo.visualize()

	return fishinfo


def create_emulator(compressor):
	chisq_em = GPREmulator(compressor=compressor)

	chisq_em.validate(make_plot=True)
	chisq_em.fit()

	# Pickle the Emulator
	dump(chisq_em, f'emulators/{type(compressor).__name__}_GPREmulator.joblib')

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


def run_mcmc(emulator, data_vector, p0, nwalkers=100, burn_in_steps=100, nsteps=2500, truths=None, llhood='gauss'):
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
		sampler.run_mcmc(state, nsteps, progress=True)

		print('Generating corner plot')
		# Make corner plot
		flat_samples = sampler.get_chain(discard=burn_in_steps, thin=15, flat=True)

		fig = corner.corner(
			flat_samples, labels=cosmologies.get_cosmological_parameters('fid').columns[1:], truths=truths
		)

		fig.savefig('plots/corner.png')


def test():
	res = {
		'type': [],  # chisq or fisherinfo, type of compressor
		'min_det': [],
		'increase': [],  # chisq_increase or fisher_info_increase
		'final_crosscorr_det': [],
		'vector_length': [],
	}

	slics_pds, cosmoslics_pds, dist_powers = run()

	for min_det in list(np.linspace(.1, .9, 5)): # list(np.logspace(-11, -3, 5)) + 

		for chisq_inc in [.01, .1, .2, .5]:
			plots_dir = f'plots/plots_det{min_det:.1e}_chisq{chisq_inc}'
			os.mkdir(plots_dir)
			c = create_chisq_comp(slics_pds, cosmoslics_pds, dist_powers, chisq_inc, min_det, plots_dir=plots_dir)

			res['type'].append('chisq')
			res['min_det'].append(min_det)
			res['increase'].append(chisq_inc)
			res['final_crosscorr_det'].append(np.linalg.det(c.slics_crosscorr_matrix))
			res['vector_length'].append(c.data_vector_length)
	
			df = pandas.DataFrame(res)
			df.to_csv('plots/test_run.csv', index=False)
		
		for fishinfo_inc in [.005, .02, .05, .1]:
			plots_dir = f'plots/plots_det{min_det:.1e}_fishinfo{fishinfo_inc}'
			os.mkdir(plots_dir)
			c = create_fishinfo_comp(slics_pds, cosmoslics_pds, dist_powers, fishinfo_inc, min_det, plots_dir=plots_dir)

			res['type'].append('fishinfo')
			res['min_det'].append(min_det)
			res['increase'].append(fishinfo_inc)
			res['final_crosscorr_det'].append(np.linalg.det(c.slics_crosscorr_matrix))
			res['vector_length'].append(c.data_vector_length)
	
			df = pandas.DataFrame(res)
			df.to_csv('plots/test_run.csv', index=False)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(prog='KiDS analysis pipeline')
	parser.add_argument('-l', '--load', action='store_true', help='Flag to set to load from pickle or not. Passed flag means load pickle object')
	parser.add_argument('-p', '--pickle-path', type=str, help='Path to Emulator pickle object', default='emulators/all_regions_ChiSq_GPR_Emulator.joblib')

	parser.add_argument('--n-walkers', type=int, help='Number of MCMC walkers', default=500)
	parser.add_argument('--burn-in-steps', type=int, default=1000, help='Number of burn in steps')
	parser.add_argument('--n-steps', type=int, default=10000, help='Number of MCMC steps (not including burn in)')
	parser.add_argument('--likelihood', type=str, default='sellentin-heavens', help='Likelihood function to use')

	parser.add_argument('-t', '--test', action='store_true', help='Run test function')

	args = parser.parse_args()

	if args.test:
		print('Running test')
		test()
		sys.exit()

	if not args.load:
		print('Loading data')
		slics_pds, cosmoslics_pds, dist_powers = run()
		comp = create_chisq_comp(slics_pds, cosmoslics_pds, dist_powers, chisq_increase=0.01, minimum_crosscorr_det=.9)
		emu = create_emulator(comp)
	else:
		print('Loading pickle file', args.pickle_path)
		emu = run_with_pickle(args.pickle_path)

	print(f'Running MCMC with nwalkers={args.n_walkers}, burn_in_steps={args.burn_in_steps}, nsteps={args.n_steps}, llhood={args.likelihood}')
	run_mcmc(emu, emu.compressor.avg_slics_data_vector, p0=np.random.rand(4), truths=slics_truths, 
			nwalkers=args.n_walkers, burn_in_steps=args.burn_in_steps, nsteps=args.n_steps, llhood=args.likelihood)
	