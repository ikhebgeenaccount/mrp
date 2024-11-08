from typing import List

import numpy as np

from analysis.cosmology_data import CosmologyData
from analysis.data_compression.compressor import Compressor
from analysis.data_compression.criteria.criterium import Criterium
from analysis.data_compression.index_compressor import IndexCompressor
from analysis.persistence_diagram import BaseRangedMap, PersistenceDiagram
from utils.is_notebook import is_notebook

if is_notebook():
	from tqdm.notebook import tqdm
else:
	from tqdm import tqdm


class GrowingVectorCompressor(IndexCompressor):

	def __init__(
			self, cosmoslics_datas: List[CosmologyData], slics_data: List[CosmologyData],
			criterium: Criterium,
			max_data_vector_length: int, minimum_feature_count: float=0, correlation_determinant_criterium: Criterium=None,
			stop_after_n_unaccepted: float=np.inf, add_feature_count=False, verbose=False
		):
		super().__init__(cosmoslics_datas, slics_data, indices=[], add_feature_count=add_feature_count)
		self.map_indices = None

		self.criterium = criterium

		self.max_data_vector_length = max_data_vector_length

		self.minimum_feature_count = minimum_feature_count
		self.correlation_det_crit = correlation_determinant_criterium

		self.stop_after_n_unaccepted = stop_after_n_unaccepted

		self.verbose = verbose

		self._setup_test_indices(cosmoslics_datas, slics_data)


	def _setup_test_indices(self, cosmoslics_datas, slics_data):
		self.pixel_scores = self.criterium.pixel_scores()
		self.pixel_scores_shape = self.pixel_scores.shape
		self.pixel_scores_argsort = np.argsort(self.pixel_scores, axis=None)[::-1]
		# We need to be able to filter which indices have > minimum_feature_count
		# So, we build one large array of (cosmologies, zbins, dim, bng_resolution, bng_resolution)
		# Then, we can np.max over axis=0 (cosmologies) to find which pixels adhere to > minimum_feature_count
		# And filter out all others
		feature_counts = [[[cdata.dimension_pairs_count_avg[zbin][dim] * cdata.zbins_bngs_avg[zbin][dim]._transform_map() for dim in [0, 1]] for zbin in self.zbins] for cdata in cosmoslics_datas]
		# Axis=0 is the outermost [] in the list comprehension above
		self.max_feature_count = np.max(feature_counts, axis=0)

		# Filter pixel coordinates based on minimum_feature_count
		self.pixel_scores_argsort = self.pixel_scores_argsort[self.max_feature_count.flatten()[self.pixel_scores_argsort] >= self.minimum_feature_count]

		# Find first non-nan value
		self.start_index = self._find_first_nonnan()
		print('First index', np.unravel_index(self.pixel_scores_argsort[self.start_index], self.pixel_scores_shape))

		# Build set of indices to test
		self.test_indices = self.pixel_scores_argsort[self.start_index:]

	def _find_first_nonnan(self, from_index: int=0):
		for i in range(len(self.pixel_scores_argsort) - from_index):
			if np.isfinite(self.pixel_scores[np.unravel_index(self.pixel_scores_argsort[i + from_index], self.pixel_scores_shape)]):
				break
		return i + from_index
	
	def _get_test_indices(self):
		for ind in self.test_indices:
			yield ind

	def _build_training_set(self, cosm_datas: List[CosmologyData]):
		if self.map_indices is not None:
			return super()._build_training_set(cosm_datas)

		self.map_indices = []

		self.last_i_accepted = 0

		for i, new_index in enumerate(tqdm(self._get_test_indices(), leave=False, total=len(self.test_indices) - self.start_index)):
			if self._check_stopping_conditions(i):
				break
							
			new_unrav = np.unravel_index(new_index, self.pixel_scores_shape)
			temp_map_indices = self.map_indices + [new_unrav]

			# self.debug(f'Testing index {new_unrav}')

			# Check if we have > min_count features in this index for at least one cosmoSLICS
			if self.max_feature_count[new_unrav] < self.minimum_feature_count:
				self.debug('Minimum feature count not reached')
				self.debug('Shouldnt see this')
				continue
			
			# try:
			# The first index is always valid to add, no need to calculate anything
			# if len(self.map_indices) > 0:

			temp_compressor = IndexCompressor(self.cosmoslics_datas, self.slics_data, temp_map_indices)
			temp_compressor.compress()

			if not self._test_corr_det(temp_compressor):
				continue

			if self.criterium.acceptance_func(temp_compressor):
				self._accept_index(i, new_unrav)
				
			# except np.linalg.LinAlgError:
			# 	self.debug('np.linalg.LinAlgError')
			# 	pass

		self._print_result()

		# Set indices to be map_indices
		return self._finalize_build(cosm_datas)
	
	def _test_corr_det(self, compressor: Compressor):
		compressor._build_crosscorr_matrix()
		if not self.correlation_det_crit.acceptance_func(compressor):
			self.debug('Correlation determinant criterium not passed')
			return False
		return True	

	def _check_stopping_conditions(self, i):
		# Check if stopping conditions have been met
		if i - self.last_i_accepted >= self.stop_after_n_unaccepted:
			tqdm.write(f'Last accepted index {self.last_i_accepted}, current index {i}, stopping')
			return True

		if len(self.map_indices) == self.max_data_vector_length:
			tqdm.write('Maximum data vector length reached')
			return True
		
		return False
		
	def _accept_index(self, i, acc_index):		
		self.map_indices.append(acc_index)
		tqdm.write(f'Accepting index {i} (len = {len(self.map_indices)}): {acc_index}')

		self.last_i_accepted = i

	def _finalize_build(self, cosm_datas):
		self.set_indices(self.map_indices)
		tset = super()._build_training_set(cosm_datas)
		tset['name'] = f'{type(self).__name__}_{type(self.criterium).__name__}'
		return tset

	def _print_result(self):
		print('Resulting length data vector:', len(self.map_indices))
		print('Indices:', self.map_indices)
		print('Specifically, using:')
		for ind in self.map_indices:
			zbin_ind = ind[0]
			print(f'\t{self.zbins[zbin_ind]}: {ind[1:]}')

	def visualize(self, save=True, moments=[]):
		for iz, (zbin, zbin_lbl) in enumerate(zip(self.zbins, self.zbins_labels)):
			for dim in [0, 1]:

				x_ind_dim = self.indices[(self.indices[:, 0] == iz) * (self.indices[:, 1] == dim)][:, 3]
				y_ind_dim = self.indices[(self.indices[:, 0] == iz) * (self.indices[:, 1] == dim)][:, 2]
				
				r = [-.05, .05]
				pix_sc_map = BaseRangedMap(self.pixel_scores[iz][dim], x_range=r, y_range=r, dimension=dim, name='pixel_scores')

				fig, ax = pix_sc_map.plot(title=f'Pixel scores zbin={zbin_lbl}, dim={dim}', scatter_points=[x_ind_dim, y_ind_dim],
								scatters_are_index=True, heatmap_scatter_points=False)
				
				if save:
					self._save_plot(fig, f'visualize_pixel_scores_zbin{zbin}_dim{dim}')

		crit_fig = self.criterium.plot()
		if save:
			self._save_plot(crit_fig, f'{type(self.criterium).__name__}')
		return super().visualize(save, moments=moments)

	def _regen_crit_values(self):
		for n in range(len(self.indices)):
			temp = IndexCompressor(self.cosmoslics_datas, self.slics_data, indices=self.indices[:n + 1], add_feature_count=self.add_feature_count)
			temp.compress()
			self.criterium.acceptance_func(temp)