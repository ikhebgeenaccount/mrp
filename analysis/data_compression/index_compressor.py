from analysis.data_compression.compressor import Compressor
from analysis.persistence_diagram import BettiNumbersGrid, PersistenceDiagram

import numpy as np
from scipy.stats import moment

from typing import List


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
			training_set['target'].append(pds_bngs_merged[self.indices_t[0], self.indices_t[1], self.indices_t[2]].flatten())

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