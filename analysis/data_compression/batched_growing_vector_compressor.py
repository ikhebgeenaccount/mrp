from typing import List

import numpy as np

from analysis.cosmology_data import CosmologyData
from analysis.data_compression.compressor import Compressor
from analysis.data_compression.criteria.criterium import Criterium
from analysis.data_compression.growing_vector_compressor import GrowingVectorCompressor
from analysis.data_compression.index_compressor import IndexCompressor
from utils.is_notebook import is_notebook

if is_notebook():
	from tqdm.notebook import tqdm
else:
	from tqdm import tqdm


class BatchedGrowingVectorCompressor(GrowingVectorCompressor):

	def __init__(
		self, cosmoslics_datas: List[CosmologyData], slics_data: List[CosmologyData], criterium: Criterium,
		max_data_vector_length: int, minimum_feature_count: float=0, minimum_crosscorr_det: float=1e-5,
		batch_size=100, add_feature_count=False, verbose=False
	):
		self.batch_size = batch_size

		super().__init__(
			cosmoslics_datas, slics_data, 
			criterium=criterium,
			max_data_vector_length=max_data_vector_length,
			minimum_feature_count=minimum_feature_count,
			minimum_crosscorr_det=minimum_crosscorr_det,
			add_feature_count=add_feature_count,
			verbose=verbose
		)
	
	def _build_training_set(self, cosm_datas: List[CosmologyData]):
		if self.map_indices is not None:
			return super()._build_training_set(cosm_datas)

		self.map_indices = []

		curr_index = self.start_index
		for i in range(self.max_data_vector_length):

			# Create Compressors for batch_size indices from curr_index
			compare_values = []
			for j in range(self.batch_size):
				# Get the jth index out from curr_index
				curr_unrav = np.unravel_index(self.test_indices[curr_index + j], self.pixel_scores_shape)
				temp_comp = IndexCompressor(self.cosmoslics_datas, self.slics_data, indices=self.map_indices + [curr_unrav])

				# Check that correlation determinant is > minimum_crosscorr_det
				if not self._test_corr_det(temp_comp):
					continue

				# Append the value obtained from criterium for this appended index
				compare_values.append(self.criterium.criterium_value(temp_comp))
			
			if len(compare_values) == 0:
				raise ValueError(f'No suitable indices were found in batch of size {self.batch_size}')

			# Find the index corresponding to the highest value
			best_index = np.argmax(compare_values)

			self._accept_index(curr_index + best_index, np.unravel_index(self.test_indices[curr_index + best_index], self.pixel_scores_shape))

			curr_index = self._set_new_start(curr_index, best_index)
			
	def _set_new_start(self, curr_index, best_index):
		# Next batch will be calculated from the index following best_index
		return curr_index + best_index + 1
