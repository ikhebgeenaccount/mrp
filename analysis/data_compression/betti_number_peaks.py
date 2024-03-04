from typing import List

import numpy as np
from scipy.stats import moment
from analysis.data_compression.compressor import Compressor
from analysis.persistence_diagram import PersistenceDiagram, PixelDistinguishingPowerMap, BettiNumbersGrid

class BettiNumberPeaksCompressor(Compressor):

	def __init__(self, cosmoslics_pds: List[PersistenceDiagram], slics_pds: List[PersistenceDiagram], 
			  pixel_distinguishing_power: List[PixelDistinguishingPowerMap], min_count: int=0):
		self.pixel_distinguishing_power = pixel_distinguishing_power
		self.min_count = min_count
		super().__init__(cosmoslics_pds, slics_pds)

	def _build_training_set(self, pds: List[PersistenceDiagram]):
		# Build training set with betti number peaks
		self.dim_best_pixels = {}
		training_set_peaks = {
			'name': 'betti_number_peaks',
			'input': [],
			'target_0': [],
			'target_1': []
		}

		for dim in [0, 1]:
			# print(f'Dimension: {dim}')
			locmax, scatters = find_local_maxima(self.pixel_distinguishing_power[dim])

			# print(self.pixel_distinguishing_power[dim].map[locmax[0], locmax[1]])

			# self.pixel_distinguishing_power[dim].save_figure(os.path.join('plots', 'pixel_distinguishing_power'), scatter_points=scatters, save_name='peaks')

			# Check if all peaks have a peak count of >= min_count
			x_ind = [] if self.min_count > 0 else locmax[1]
			y_ind = [] if self.min_count > 0 else locmax[0]
			if self.min_count > 0:
				for x, y in zip(locmax[1], locmax[0]):
					passes = False
					for perdi in self.cosmoslics_pds:
						if perdi.betti_numbers_grids[dim]._transform_map()[y, x] >= self.min_count / len(perdi.dimension_pairs[dim]):
							passes = True
							break				

					if passes:
						x_ind.append(x)
						y_ind.append(y)

			self.dim_best_pixels[dim] = {
				'x_ind': x_ind,
				'y_ind': y_ind,
				'scatters': scatters
			}

		for cpd in pds:
			training_set_peaks['input'].append([val for key, val in cpd.cosm_parameters.items() if key != 'id'])
			for dim in [0, 1]:
				training_set_peaks[f'target_{dim}'].append(
					np.concatenate((
						[len(cpd.dimension_pairs[dim]) / cpd.maps_count],  # Number of features of dim
						cpd.betti_numbers_grids[dim]._transform_map()[self.dim_best_pixels[dim]['y_ind'], self.dim_best_pixels[dim]['x_ind']]
					))
				)

		# Merge dimensions in target data
		training_set_peaks['target'] = np.array([np.concatenate((t0, t1)) for t0, t1 in zip(training_set_peaks['target_0'], training_set_peaks['target_1'])])
		training_set_peaks['input'] = np.array(training_set_peaks['input'])

		return training_set_peaks
	
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
	
	def visualize(self):
		for dim in [0, 1]:
			pix_dist_map = self.pixel_distinguishing_power[dim]

			fig, ax = pix_dist_map.plot(title=f'dim={dim}', scatter_points=[self.dim_best_pixels[dim]['x_ind'], self.dim_best_pixels[dim]['y_ind']],
							   scatters_are_index=True)
			
			self._add_data_vector_labels(ax, dim)

			for mom in [1, 2, 3, 4]:
				avg_bng_cosmoslics_dim = BettiNumbersGrid(
					moment([cpd.betti_numbers_grids[dim].map for cpd in self.cosmoslics_pds], moment=mom, axis=0, nan_policy='omit', center=0 if mom == 1 else None), 
					birth_range=self.cosmoslics_pds[0].betti_numbers_grids[dim].x_range,
					death_range=self.cosmoslics_pds[0].betti_numbers_grids[dim].y_range,
					dimension=dim
				)

				fig, ax = avg_bng_cosmoslics_dim.plot(scatter_points=[self.dim_best_pixels[dim]['x_ind'], self.dim_best_pixels[dim]['y_ind']],
								scatters_are_index=True)
				self._add_data_vector_labels(ax, dim)
				ax.set_title(f'moment={mom}')



def find_local_maxima(rmap: PixelDistinguishingPowerMap, size=5):
	image = rmap._transform_map()
	image_shifts = []

	image_nonan = np.nan_to_num(image)

	for shift in range(1, size+1):
		for dir in [[1, 1], [1, 0], [0, 1], [-1, 1]]:
			for t in [-1, 1]:
				fsh = t * np.array(dir)
				shifted_image = np.roll(image_nonan, axis=(0, 1), shift=(shift * fsh[0], shift * fsh[1]))
				image_shifts.append(shifted_image)
	
	local_maxima = np.prod(np.greater(image, image_shifts), axis=0)
	local_maxima_indices = np.where(local_maxima)
	x_ind = local_maxima_indices[1]
	y_ind = local_maxima_indices[0]
	scatter_points = [rmap.get_axis_values('x')[x_ind], rmap.get_axis_values('y')[y_ind]]

	return local_maxima_indices, scatter_points