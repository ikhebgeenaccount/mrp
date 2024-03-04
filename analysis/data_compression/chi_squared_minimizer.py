from typing import List

import numpy as np
from scipy.stats import moment

from analysis.data_compression.compressor import Compressor
from analysis.persistence_diagram import BettiNumbersGrid, PersistenceDiagram, PixelDistinguishingPowerMap


class IndexCompressor(Compressor):

	def __init__(self, cosmoslics_pds: List[PersistenceDiagram], slics_pds: List[PersistenceDiagram],
			  indices):
		self.set_indices(indices)
		super().__init__(cosmoslics_pds, slics_pds)

	def set_indices(self, indices):
		self.indices = np.array(indices)
		self.indices_t = self.indices.T

	def _build_training_set(self, pds: List[PersistenceDiagram]):
		training_set = {
			'name': 'index',
			'input': [],
			'target': []
		}
		for pd in pds:
			pds_bngs_merged = np.array([pd.betti_numbers_grids[0]._transform_map(), pd.betti_numbers_grids[1]._transform_map()])
			training_set['input'].append(np.array([val for key, val in pd.cosm_parameters.items() if key != 'id']))
			training_set['target'].append(pds_bngs_merged[self.indices_t[0], self.indices_t[1], self.indices_t[2]])

		return training_set
	
	def visualize(self):
		for dim in [0, 1]:
			pix_dist_map = self.dist_powers[dim]

			x_ind_dim = self.indices[self.indices[:, 0] == dim][:, 2]
			y_ind_dim = self.indices[self.indices[:, 0] == dim][:, 1]

			fig, ax = pix_dist_map.plot(title=f'dim={dim}', scatter_points=[x_ind_dim, y_ind_dim],
							   scatters_are_index=True, heatmap_scatter_points=False)
			
			# self._add_data_vector_labels(ax, dim)

			for mom in [1, 2, 3, 4]:
				avg_bng_cosmoslics_dim = BettiNumbersGrid(
					moment([cpd.betti_numbers_grids[dim].map for cpd in self.cosmoslics_pds], moment=mom, axis=0, nan_policy='omit', center=0 if mom == 1 else None), 
					birth_range=self.cosmoslics_pds[0].betti_numbers_grids[dim].x_range,
					death_range=self.cosmoslics_pds[0].betti_numbers_grids[dim].y_range,
					dimension=dim
				)

				fig, ax = avg_bng_cosmoslics_dim.plot(scatter_points=[x_ind_dim, y_ind_dim],
								scatters_are_index=True)
				# self._add_data_vector_labels(ax, dim)
				ax.set_title(f'moment={mom}')
	
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
	

class ChiSquaredMinimizer(IndexCompressor):

	def __init__(self, cosmoslics_pds: List[PersistenceDiagram], slics_pds: List[PersistenceDiagram],
			  dist_powers: List[PixelDistinguishingPowerMap], max_data_vector_length=250, minimum_feature_count=0):
		self.map_indices = None
		self.dist_powers = dist_powers
		self.dist_powers_merged = np.array([dist_powers[0]._transform_map(), dist_powers[1]._transform_map()])
		self.dist_powers_shape = self.dist_powers_merged.shape
		self.dist_powers_argsort = np.argsort(self.dist_powers_merged, axis=None)[::-1]

		self.max_data_vector_length = max_data_vector_length

		self.minimum_feature_count = minimum_feature_count

		feature_counts = [[len(cpd.dimension_pairs[dim]) * cpd.betti_numbers_grids[0]._transform_map() for cpd in cosmoslics_pds] for dim in [0, 1]]
		self.smallest_feature_count = np.min(feature_counts, axis=1)

		# Find first non-nan value
		for i in range(len(self.dist_powers_argsort)):
			if np.isfinite(self.dist_powers_merged[np.unravel_index(self.dist_powers_argsort[i], self.dist_powers_shape)]):
				break
		self.start_index = i

		super().__init__(cosmoslics_pds, slics_pds, indices=[])

	def _build_training_set(self, pds: List[PersistenceDiagram]):
		if self.map_indices is not None:
			return super()._build_training_set(pds)

		# Build set of indices to test
		test_indices = self.dist_powers_argsort[self.start_index:]

		self.map_indices = []
		self.chisq_values = []

		prev_chisq = 0
		# Set chi_sq to start at 1 to make sure first index is added to vector
		chi_sq = 1

		for i, new_index in enumerate(test_indices):
			new_unrav = np.unravel_index(new_index, self.dist_powers_shape)
			temp_map_indices = self.map_indices + [new_unrav]

			# Check if we have > min_count features in this index for at least one cosmoSLICS
			if self.smallest_feature_count[new_unrav] < self.minimum_feature_count:
				continue
			
			try:
				# The first index is always valid to add, no need to calculate anything
				if len(self.map_indices) > 0:

					temp_compressor = IndexCompressor(self.cosmoslics_pds, self.slics_pds, temp_map_indices)

					# Calculate chi squared value
					sub = (temp_compressor.avg_slics_data_vector - temp_compressor.cosmoslics_training_set['target'])
					intermed = np.matmul(sub, np.linalg.inv(temp_compressor.slics_covariance_matrix))
					chi_sq = (1. / 26.) * np.sum(np.matmul(intermed, sub.T))

				if chi_sq - prev_chisq > .2:
					# print(f'Accepting {i}th index')
					self.map_indices.append(new_unrav)
					self.chisq_values.append(chi_sq)

					prev_chisq = chi_sq
			except np.linalg.LinAlgError:
				pass

			if len(self.map_indices) == self.max_data_vector_length:
				break

		print('Resulting length data vector:', len(self.map_indices))

		# Set indices to be map_indices
		self.set_indices(self.map_indices)
		tset = super()._build_training_set(pds)
		tset['name'] = 'chi_sq_minimizer'
		return tset

		