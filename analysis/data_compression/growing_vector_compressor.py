from typing import List

import numpy as np

from analysis.data_compression.compressor import Compressor
from analysis.data_compression.index_compressor import IndexCompressor
from analysis.persistence_diagram import PersistenceDiagram


class GrowingVectorCompressor(IndexCompressor):

	def __init__(
			self, cosmoslics_pds: List[PersistenceDiagram], slics_pds: List[PersistenceDiagram], pixel_scores: np.ndarray,
			max_data_vector_length: int, minimum_feature_count: float=0, minimum_crosscorr_det: float=1e-5,
			stop_after_n_unaccepted: float=np.inf, verbose=False
		):
		self.map_indices = None
		self.pixel_scores = pixel_scores
		self.pixel_scores_shape = self.pixel_scores.shape
		self.pixel_scores_argsort = np.argsort(self.pixel_scores, axis=None)[::-1]

		self.max_data_vector_length = max_data_vector_length

		self.minimum_feature_count = minimum_feature_count

		feature_counts = [[len(cpd.dimension_pairs[dim]) * cpd.betti_numbers_grids[0]._transform_map() for cpd in cosmoslics_pds] for dim in [0, 1]]
		self.max_feature_count = np.max(feature_counts, axis=1)

		# Find first non-nan value
		for i in range(len(self.pixel_scores_argsort)):
			if np.isfinite(self.pixel_scores[np.unravel_index(self.pixel_scores_argsort[i], self.pixel_scores_shape)]):
				break
		self.start_index = i

		self.min_crosscorr_det = minimum_crosscorr_det

		self.stop_after_n_unaccepted = stop_after_n_unaccepted

		self.verbose = verbose

		super().__init__(cosmoslics_pds, slics_pds, indices=[])

	def acceptance_func(self, compressor: Compressor):
		raise NotImplementedError('Subclasses of VectorGrowthCompressor must implement acceptance_func')

	def _build_training_set(self, pds: List[PersistenceDiagram]):
		if self.map_indices is not None:
			return super()._build_training_set(pds)

		# Build set of indices to test
		test_indices = self.pixel_scores_argsort[self.start_index:]

		self.map_indices = []

		last_i_accepted = 0

		for i, new_index in enumerate(test_indices):
			new_unrav = np.unravel_index(new_index, self.pixel_scores_shape)
			temp_map_indices = self.map_indices + [new_unrav]

			# self.debug(f'Testing index {new_unrav}')

			# Check if we have > min_count features in this index for at least one cosmoSLICS
			if self.max_feature_count[new_unrav] < self.minimum_feature_count:
				self.debug('Minimum feature count not reached')
				continue
			
			try:
				# The first index is always valid to add, no need to calculate anything
				# if len(self.map_indices) > 0:

				temp_compressor = IndexCompressor(self.cosmoslics_pds, self.slics_pds, temp_map_indices)

				temp_compressor._build_crosscorr_matrix()
				if np.linalg.det(temp_compressor.slics_crosscorr_matrix) < self.min_crosscorr_det:
					continue

				if self.acceptance_func(temp_compressor):
					self.map_indices.append(new_unrav)
					self.debug(f'Accepting index {new_unrav}')

					last_i_accepted = i
				
			except np.linalg.LinAlgError:
				self.debug('np.linalg.LinAlgError')
				pass

			if len(self.map_indices) == self.max_data_vector_length:
				self.debug('Maximum data vector length reached')
				break

			if i - last_i_accepted >= self.stop_after_n_unaccepted:
				self.debug(f'Last accepted index {last_i_accepted}, current index {i}, stopping')
				break

		print('Resulting length data vector:', len(self.map_indices))

		# Set indices to be map_indices
		self.set_indices(self.map_indices)
		tset = super()._build_training_set(pds)
		tset['name'] = 'chi_sq_minimizer'
		return tset