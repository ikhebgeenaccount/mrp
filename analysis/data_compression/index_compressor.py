from analysis.cosmology_data import CosmologyData
from analysis.data_compression.compressor import Compressor
from analysis.persistence_diagram import BaseRangedMap, BettiNumbersGrid, PersistenceDiagram

import numpy as np
from scipy.stats import moment

from typing import List


class IndexCompressor(Compressor):

	def __init__(self, cosmoslics_datas: List[CosmologyData], slics_data: List[CosmologyData],
			  indices, add_feature_count: bool=False):
		self.set_indices(indices)
		self.add_feature_count = add_feature_count
		super().__init__(cosmoslics_datas, slics_data)

	def set_indices(self, indices):
		self.indices = np.array(indices)
		self.indices_t = self.indices.T

	def _build_training_set(self, cosm_datas: List[CosmologyData]):
		training_set = {
			'name': 'index',
			'input': [],
			'target': []
		}
		for cosmdata in cosm_datas:
			bngs = []
			dimpair_counts = []
			for zbin in self.zbins:
				# Add all redshift bins' BettiNumbersGrids (both dimensions)
				bngs.append(
					[cosmdata.zbins_bngs_avg[zbin][0]._transform_map(),
					cosmdata.zbins_bngs_avg[zbin][1]._transform_map()]
				)

				dimpair_counts += list(cosmdata.dimension_pairs_count_avg[zbin])

			# Turn list into numpy array for easy slicing
			bngs_merged = np.array(bngs)
			self._select_pixels(bngs_merged, dimpair_counts, training_set)

			training_set['input'].append(np.array([val for key, val in cosmdata.cosm_parameters.items() if key != 'id']))

		return training_set

	def _select_pixels(self, bngs_merged, dimpair_counts, training_set):		
		
		if len(self.indices) > 0 and self.add_feature_count:
			training_set['target'].append(
				np.concatenate(
					(dimpair_counts, 
					bngs_merged[self.indices_t[0], self.indices_t[1], self.indices_t[2], self.indices_t[3]].flatten())
				)
			)
		elif self.add_feature_count:
			training_set['target'].append(dimpair_counts)
		else:
			training_set['target'].append(bngs_merged[self.indices_t[0], self.indices_t[1], self.indices_t[2], self.indices_t[3]].flatten())

	def _build_slics_training_set(self, slics_data: List[CosmologyData]):
		training_set = {
			'name': 'index',
			'input': [],
			'target': []
		}
		sdata = slics_data[0]
		for index_pd in range(slics_data[0].pds_count):
			bngs = []
			dimpair_counts = []
			for zbin in self.zbins:
				# Add all redshift bins' BettiNumbersGrids (both dimensions)
				bngs.append([
					sdata.zbins_bngs[zbin][0][index_pd]._transform_map(),
					sdata.zbins_bngs[zbin][1][index_pd]._transform_map()]
				)

				dimpair_counts += list(sdata.zbins_dimension_pairs_counts[zbin][index_pd])

			# Turn list into numpy array for easy slicing
			bngs_merged = np.array(bngs)
			self._select_pixels(bngs_merged, dimpair_counts, training_set)

			training_set['input'].append(np.array([val for key, val in sdata.cosm_parameters.items() if key != 'id']))

		return training_set

	def visualize(self, save=True):
		for iz, zbin in enumerate(self.zbins):
			for dim in [0, 1]:

				x_ind_dim = self.indices[(self.indices[:, 0] == iz) * (self.indices[:, 1] == dim)][:, 3]
				y_ind_dim = self.indices[(self.indices[:, 0] == iz) * (self.indices[:, 1] == dim)][:, 2]

				# for mom in [1, 2, 3, 4]:
				# 	avg_bng_cosmoslics_dim = BettiNumbersGrid(
				# 		moment([cpd.betti_numbers_grids[dim].map for cpd in self.cosmoslics_pds], moment=mom, axis=0, nan_policy='omit', center=0 if mom == 1 else None),
				# 		birth_range=self.cosmoslics_pds[0].betti_numbers_grids[dim].x_range,
				# 		death_range=self.cosmoslics_pds[0].betti_numbers_grids[dim].y_range,
				# 		dimension=dim
				# 	)

				# 	fig, ax = avg_bng_cosmoslics_dim.plot(scatter_points=[x_ind_dim, y_ind_dim],
				# 					scatters_are_index=True)
				# 	# self._add_data_vector_labels(ax, dim)
				# 	ax.set_title(f'dim={dim}, moment={mom}')

				# 	if save:
				# 		self._save_plot(fig, f'visualize_dim{dim}_mom{mom}')

	def _add_data_vector_labels(self, ax, dim):
		if dim == 0:
			# Entry index starts at one since first entry is feature count, not plotted
			entry_index = 1
		elif dim == 1:
			entry_index = 2 + len(self.dim_best_pixels[0]['x_ind'])

		for x, y in zip(self.dim_best_pixels[dim]['x_ind'], self.dim_best_pixels[dim]['y_ind']):

			x_values = self.pixel_distinguishing_power[dim].get_axis_values('x')[x]
			y_values = self.pixel_distinguishing_power[dim].get_axis_values('y')[::-1][y]
			ax.scatter(x_values, y_values, s=3, alpha=.6, color='red')
			ax.text(x_values, y_values, str(entry_index), color='white')
			entry_index += 1