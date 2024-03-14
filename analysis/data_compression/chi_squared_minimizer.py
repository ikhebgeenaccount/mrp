from typing import List

import numpy as np
from analysis.data_compression.index_compressor import IndexCompressor

from analysis.persistence_diagram import PersistenceDiagram, PixelDistinguishingPowerMap


class ChiSquaredMinimizer(IndexCompressor):

	def __init__(self, cosmoslics_pds: List[PersistenceDiagram], slics_pds: List[PersistenceDiagram],
			  dist_powers: List[PixelDistinguishingPowerMap], max_data_vector_length=250, minimum_feature_count=0, chisq_increase=0.2, verbose=False):
		self.map_indices = None
		self.dist_powers = dist_powers
		self.dist_powers_merged = np.array([dist_powers[0]._transform_map(), dist_powers[1]._transform_map()])
		self.dist_powers_shape = self.dist_powers_merged.shape
		self.dist_powers_argsort = np.argsort(self.dist_powers_merged, axis=None)[::-1]

		self.max_data_vector_length = max_data_vector_length

		self.minimum_feature_count = minimum_feature_count

		feature_counts = [[len(cpd.dimension_pairs[dim]) * cpd.betti_numbers_grids[0]._transform_map() for cpd in cosmoslics_pds] for dim in [0, 1]]
		self.max_feature_count = np.max(feature_counts, axis=1)

		# Find first non-nan value
		for i in range(len(self.dist_powers_argsort)):
			if np.isfinite(self.dist_powers_merged[np.unravel_index(self.dist_powers_argsort[i], self.dist_powers_shape)]):
				break
		self.start_index = i

		self.chisq_increase = chisq_increase

		self.verbose = verbose

		super().__init__(cosmoslics_pds, slics_pds, indices=[])

	def _build_training_set(self, pds: List[PersistenceDiagram]):
		if self.map_indices is not None:
			return super()._build_training_set(pds)

		# Build set of indices to test
		test_indices = self.dist_powers_argsort[self.start_index:]

		self.map_indices = []
		self.chisq_values = []
		self.fisher_dets = []

		prev_chisq = 0.
		chi_sq = 0.
		fisher_det = 0.

		for i, new_index in enumerate(test_indices):
			new_unrav = np.unravel_index(new_index, self.dist_powers_shape)
			temp_map_indices = self.map_indices + [new_unrav]

			self.debug(f'Testing index {new_unrav} against prev_chisq={prev_chisq:.5f}')

			# Check if we have > min_count features in this index for at least one cosmoSLICS
			if self.max_feature_count[new_unrav] < self.minimum_feature_count:
				self.debug('Minimum feature count not reached')
				continue
			
			try:
				# The first index is always valid to add, no need to calculate anything
				# if len(self.map_indices) > 0:

				temp_compressor = IndexCompressor(self.cosmoslics_pds, self.slics_pds, temp_map_indices)

				# Calculate chi squared value
				sub = (temp_compressor.avg_slics_data_vector - temp_compressor.cosmoslics_training_set['target'])
				intermed = np.matmul(sub, np.linalg.inv(temp_compressor.slics_covariance_matrix))
				chi_sq = (1. / 26.) * np.sum(np.matmul(intermed, sub.T))
				fisher_det = np.linalg.det(temp_compressor.fisher_matrix)

				self.debug(f'chisq={chi_sq:.5f}, fisher_det={fisher_det:.5e}')

				if chi_sq - prev_chisq > self.chisq_increase:
					self.debug('Accepting index')
					self.map_indices.append(new_unrav)
					self.chisq_values.append(chi_sq)
					self.fisher_dets.append(fisher_det)

					prev_chisq = chi_sq
			except np.linalg.LinAlgError:
				self.debug('np.linalg.LinAlgError')
				pass

			if len(self.map_indices) == self.max_data_vector_length:
				self.debug('Maximum data vector length reached')
				break

		print('Resulting length data vector:', len(self.map_indices))

		# Set indices to be map_indices
		self.set_indices(self.map_indices)
		tset = super()._build_training_set(pds)
		tset['name'] = 'chi_sq_minimizer'
		return tset

		