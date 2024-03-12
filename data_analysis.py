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


pipeline = Pipeline(filter_region=1, save_plots=True, force_recalculate=True, do_remember_maps=False, bng_resolution=100, three_sigma_mask=True)
pipeline.find_max_min_values_maps(save_all_values=False, save_maps=False)
# pipeline.all_values_histogram()

pipeline.read_maps()
pipeline.calculate_variance()

slics_pds = pipeline.slics_pds
cosmoslics_pds = pipeline.cosmoslics_pds
dist_powers = pipeline.dist_powers

chisqmin = ChiSquaredMinimizer(cosmoslics_pds, slics_pds, dist_powers, max_data_vector_length=100, minimum_feature_count=40)
chisqmin._calculate_derivatives_lsq()
chisqmin._calculate_fisher_matrix()

chisqmin.plot_fisher_matrix()
chisqmin.plot_crosscorr_matrix()
chisqmin.plot_data_vectors(include_slics=True)

fig, ax = plt.subplots()
ax.set_ylabel('$\chi^2$')
ax.set_xlabel('Data vector entry')
ax.plot(chisqmin.chisq_values)
fig.savefig('plots/chisqmin_data_vector.png')

# To compare 
full_grid = FullGrid(cosmoslics_pds, slics_pds)
full_grid._calculate_derivatives_lsq()
full_grid._calculate_fisher_matrix()
full_grid.plot_fisher_matrix()
# full_grid.plot_crosscorr_matrix()