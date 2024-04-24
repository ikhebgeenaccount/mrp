from typing import List

import numpy as np
from analysis.cosmology_data import CosmologyData
from analysis.data_compression.compressor import Compressor
from analysis.data_compression.full_grid import FullGrid
from analysis.data_compression.growing_vector_compressor import GrowingVectorCompressor
from analysis.data_compression.index_compressor import IndexCompressor

from analysis.persistence_diagram import PersistenceDiagram, PixelDistinguishingPowerMap, BaseRangedMap


class FisherInfoMaximizer(GrowingVectorCompressor):

	def __init__(self, cosmoslics_datas: List[CosmologyData], slics_data: List[CosmologyData], data_vector_length, 
			  fisher_info_increase:float=0.05, minimum_crosscorr_det: float=1e-5, minimum_feature_count: float=0, verbose=False):
		self.data_vector_length = data_vector_length

		full_grid = FullGrid(cosmoslics_datas, slics_data)

		# Sort full_grid by fisher info
		# Minus to make argsort descending order
		collapsed_fisher = np.reshape(np.max(-full_grid.fisher_matrix_per_entry, axis=(0, 1)), (len(slics_data[0].zbins_pds), 2, 100, 100))

		self.fisher_info_increase = fisher_info_increase
		self.prev_fisher_info = 0.
		self.fisher_info_vals = []

		super().__init__(cosmoslics_datas, slics_data, pixel_scores=collapsed_fisher, max_data_vector_length=data_vector_length,
			minimum_feature_count=minimum_feature_count, minimum_crosscorr_det=minimum_crosscorr_det, verbose=verbose)

	def acceptance_func(self, compressor: Compressor):
		new_fisher_info = compressor.fisher_matrix[1, 1]

		if new_fisher_info / self.prev_fisher_info >= 1. + self.fisher_info_increase:
			self.prev_fisher_info = new_fisher_info

			self.fisher_info_vals.append(new_fisher_info)
			return True
		return False

	# def _build_training_set(self, pds: List[PersistenceDiagram]):
		
	# 	# Fix -inf entries to be inf again to ensure they don't jump to front of argsort
	# 	collapsed_fisher[collapsed_fisher == -np.inf] = np.inf
	# 	sorted_indices = np.argsort(collapsed_fisher)

	# 	# Create collapsed fisher maps for visualization purposes
	# 	self.collapsed_fisher_maps = []
	# 	coll_fish_resh = np.reshape(collapsed_fisher, (2, 100, 100))
	# 	for dim in [0, 1]:
	# 		x_range = pds[0].betti_numbers_grids[0].x_range
	# 		y_range = pds[0].betti_numbers_grids[0].y_range
	# 		coll_fish_map = BaseRangedMap(name='collapsed_fisher_map', dimension=dim, x_range=x_range, y_range=y_range, map=coll_fish_resh[dim])

	# 		self.collapsed_fisher_maps.append(coll_fish_map)

	# 	unrav_indices = np.unravel_index(sorted_indices[:self.data_vector_length], (2, *pds[0].betti_numbers_grids[0].map.shape))

	# 	self.set_indices(np.array(unrav_indices).T)

	# 	return super()._build_training_set(pds)
