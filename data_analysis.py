import corner
from emcee import EnsembleSampler
import seaborn as sns
import pandas as pd
import scipy
import glob
import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

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

	print('Compressing data with ChiSquaredMinimizer...')
	chisqmin = ChiSquaredMinimizer(filter_region=1, cosmoslics_pds, slics_pds, dist_powers, max_data_vector_length=100, minimum_feature_count=40, chisq_increase=0.05, verbose=True)

	print('Plotting ChiSquaredMinimizer matrices and data vector...')
	chisqmin.plot_fisher_matrix()
	chisqmin.plot_crosscorr_matrix()
	chisqmin.plot_data_vectors(include_slics=True)
	chisqmin.visualize()

	fig, ax = plt.subplots()
	ax.set_ylabel('$\chi^2$')
	ax.set_xlabel('Data vector entry')
	ax.plot(chisqmin.chisq_values)
	fig.savefig('plots/chisqmin_chisq_values.png')

	fig, ax = plt.subplots()
	ax.set_ylabel('Determinant of Fisher matrix')
	ax.set_xlabel('Data vector entry')
	ax.plot(chisqmin.fisher_dets)
	ax.semilogy()
	fig.savefig('plots/chisqmin_fisher_dets.png')

	print('Creating FullGrid Fisher matrix...')
	# To compare 
	full_grid = FullGrid(cosmoslics_pds, slics_pds)
	full_grid.plot_fisher_matrix()
	# full_grid.plot_crosscorr_matrix()

	chisq_em = GPREmulator(compressor=chisqmin)

	chisq_em.validate(make_plot=True)
	chisq_em.fit()	

	run_mcmc(chisq_em, chisqmin.avg_slics_data_vector, p0=np.random.rand(4), truths=slics_truths, nwalkers=500, nsteps=10000, llhood='sellentin-heavens')


def run_mcmc(emulator, data_vector, p0, data_vector_err=None, nwalkers=100, burn_in_steps=100, nsteps=2500, truths=None, llhood='gauss'):
	with np.errstate(invalid='ignore'):
		ndim = len(p0)

		# init_walkers = np.random.rand(nwalkers, ndim)

		mcmc_ = MCMC(emulator, data_vector, emulator.compressor.slics_covariance_matrix)

		init_walkers = mcmc_.get_random_init_walkers(nwalkers)

		ll = mcmc_.gaussian_likelihood if llhood == 'gauss' else mcmc_.sellentin_heavens_likelihood

		sampler = EnsembleSampler(nwalkers, ndim, ll)

		state = sampler.run_mcmc(init_walkers, burn_in_steps)
		sampler.reset()

		sampler.run_mcmc(state, nsteps, progress='notebook')

		# Make corner plot
		flat_samples = sampler.get_chain(discard=burn_in_steps, thin=15, flat=True)

		fig = corner.corner(
			flat_samples, labels=cosmologies.get_cosmological_parameters('fid').columns[1:], truths=truths
		)

		fig.savefig('plots/corner.png')


run()