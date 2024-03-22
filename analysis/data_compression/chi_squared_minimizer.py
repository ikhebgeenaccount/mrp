from typing import List

import numpy as np
from analysis.data_compression.index_compressor import IndexCompressor
from analysis.data_compression.growing_vector_compressor import GrowingVectorCompressor
from analysis.persistence_diagram import PersistenceDiagram, PixelDistinguishingPowerMap


class ChiSquaredMinimizer(GrowingVectorCompressor):

	def __init__(self, cosmoslics_pds: List[PersistenceDiagram], slics_pds: List[PersistenceDiagram],
			  dist_powers: List[PixelDistinguishingPowerMap], max_data_vector_length=250, minimum_feature_count=0, chisq_increase=0.2, verbose=False):
		self.dist_powers = dist_powers
		self.dist_powers_merged = np.array([dist_powers[0]._transform_map(), dist_powers[1]._transform_map()])

		self.chisq_increase = chisq_increase
		self.prev_chisq = 0.

		self.chisq_values = []
		self.fisher_dets = []

		super().__init__(cosmoslics_pds, slics_pds, pixel_scores=self.dist_powers_merged, max_data_vector_length=max_data_vector_length,
		minimum_feature_count=minimum_feature_count, verbose=verbose)	

	def acceptance_func(self, compressor: IndexCompressor):
		# Calculate chi squared value
		sub = (compressor.avg_slics_data_vector - compressor.cosmoslics_training_set['target'])
		intermed = np.matmul(sub, np.linalg.inv(compressor.slics_covariance_matrix))
		chi_sq = (1. / 26.) * np.sum(np.matmul(intermed, sub.T))
		fisher_det = np.linalg.det(compressor.fisher_matrix)

		self.debug(f'chisq={chi_sq:.5f}, fisher_det={fisher_det:.5e}')

		if chi_sq - self.prev_chisq > self.chisq_increase:
			self.debug('Accepting index')

			self.prev_chisq = chi_sq

			self.chisq_values.append(chi_sq)
			self.fisher_dets.append(fisher_det)

			return True
		return False

	# def _build_training_set(self, pds: List[PersistenceDiagram]):
	# 	if self.map_indices is not None:
	# 		return super()._build_training_set(pds)

	# 	# Build set of indices to test
	# 	test_indices = self.dist_powers_argsort[self.start_index:]

	# 	self.map_indices = []
	# 	self.chisq_values = []
	# 	self.fisher_dets = []

	# 	prev_chisq = 0.
	# 	chi_sq = 0.
	# 	fisher_det = 0.

	# 	for i, new_index in enumerate(test_indices):
	# 		new_unrav = np.unravel_index(new_index, self.dist_powers_shape)
	# 		temp_map_indices = self.map_indices + [new_unrav]

	# 		self.debug(f'Testing index {new_unrav} against prev_chisq={prev_chisq:.5f}')

	# 		# Check if we have > min_count features in this index for at least one cosmoSLICS
	# 		if self.max_feature_count[new_unrav] < self.minimum_feature_count:
	# 			self.debug('Minimum feature count not reached')
	# 			continue
			
	# 		try:
	# 			# The first index is always valid to add, no need to calculate anything
	# 			# if len(self.map_indices) > 0:

	# 			temp_compressor = IndexCompressor(self.cosmoslics_pds, self.slics_pds, temp_map_indices)

	# 			# Calculate chi squared value
	# 			sub = (temp_compressor.avg_slics_data_vector - temp_compressor.cosmoslics_training_set['target'])
	# 			intermed = np.matmul(sub, np.linalg.inv(temp_compressor.slics_covariance_matrix))
	# 			chi_sq = (1. / 26.) * np.sum(np.matmul(intermed, sub.T))
	# 			fisher_det = np.linalg.det(temp_compressor.fisher_matrix)

	# 			self.debug(f'chisq={chi_sq:.5f}, fisher_det={fisher_det:.5e}')

	# 			if chi_sq - prev_chisq > self.chisq_increase:
	# 				self.debug('Accepting index')
	# 				self.map_indices.append(new_unrav)
	# 				self.chisq_values.append(chi_sq)
	# 				self.fisher_dets.append(fisher_det)

	# 				prev_chisq = chi_sq
	# 		except np.linalg.LinAlgError:
	# 			self.debug('np.linalg.LinAlgError')
	# 			pass

	# 		if len(self.map_indices) == self.max_data_vector_length:
	# 			self.debug('Maximum data vector length reached')
	# 			break

	# 	print('Resulting length data vector:', len(self.map_indices))

	# 	# Set indices to be map_indices
	# 	self.set_indices(self.map_indices)
	# 	tset = super()._build_training_set(pds)
	# 	tset['name'] = 'chi_sq_minimizer'
	# 	return tset

		