

from typing import List

import numpy as np
from analysis.cosmology_data import CosmologyData
from analysis.data_compression.batched_growing_vector_compressor import BatchedGrowingVectorCompressor
from analysis.data_compression.criteria.criterium import Criterium


class BatchedGrowingVectorPerZbinCompressor(BatchedGrowingVectorCompressor):

	def __init__(
		self, cosmoslics_datas: List[CosmologyData], slics_data: List[CosmologyData], criterium: Criterium,
		data_vector_length_per_zbin: int, minimum_feature_count: float=0, correlation_determinant_criterium: Criterium=None,
		batch_size=100, add_feature_count=False, verbose=False
	):
		self.data_vector_length_per_zbin = data_vector_length_per_zbin
		max_data_vector_length = len(slics_data[0].zbins) * data_vector_length_per_zbin
		super().__init__(
			cosmoslics_datas, slics_data, criterium,
			max_data_vector_length=max_data_vector_length,
			minimum_feature_count=minimum_feature_count,
			correlation_determinant_criterium=correlation_determinant_criterium,
			batch_size=batch_size,
			add_feature_count=add_feature_count,
			verbose=verbose
		)

	def _setup_test_indices(self, cosmoslics_datas, slics_data):
		self.pixel_scores = self.criterium.pixel_scores()
		self.pixel_scores_shape = self.pixel_scores.shape

		feature_counts = [[[cdata.dimension_pairs_count_avg[zbin][dim] * cdata.zbins_bngs_avg[zbin][dim]._transform_map() for dim in [0, 1]] for zbin in self.zbins] for cdata in cosmoslics_datas]
		# Axis=0 is the outermost [] in the list comprehension above
		self.max_feature_count = np.max(feature_counts, axis=0)

		self.pixel_scores_argsort = []
		self.pixel_count_per_zbin = []
		# Instead of sorting all zbins at the same time, we need to sort each zbin separately
		for i, zbin in enumerate(slics_data[0].zbins):
			pixarg = np.argsort(self.pixel_scores[i], axis=None)[::-1]

			# Filter pixels which have feature count < minimum feature count
			pixarg = pixarg[self.max_feature_count[i].flatten()[pixarg] >= self.minimum_feature_count] +\
				  i * 2 * int(np.square(slics_data[0].zbins_bngs_avg[zbin][0].map.shape[0]))
			self.pixel_scores_argsort += list(pixarg)
			self.pixel_count_per_zbin.append(len(pixarg))

		self.curr_zbin = 0
		self.curr_zbin_count = 0
		self.start_index = self._find_first_nonnan()	

		print(f'{self.pixel_count_per_zbin=}')	
		print(f'{self.start_index=}')

	def _accept_index(self, i, acc_index):
		self.curr_zbin_count += 1
		return super()._accept_index(i, acc_index)
	
	def _set_new_start(self, curr_index, best_index):
		# Once one zbin has data_vector_length_per_zbin entries, skip to the next zbin
		if self.curr_zbin * self.data_vector_length_per_zbin + self.curr_zbin_count == self.max_data_vector_length:
			# This was the last data vector entry, iteration will stop
			return -1
		elif self.curr_zbin_count == self.data_vector_length_per_zbin:
			self.curr_zbin += 1
			self.curr_zbin_count = 0

			# New index is the start of the next zbin
			new_start = sum(self.pixel_count_per_zbin[:self.curr_zbin])
			return self._find_first_nonnan(new_start)
		else:
			return super()._set_new_start(curr_index, best_index)