from typing import List

import numpy as np
from analysis.data_compression.compressor import Compressor
from analysis.persistence_diagram import PersistenceDiagram, PixelDistinguishingPowerMap

class BettiNumberPeaksCompressor(Compressor):

	def __init__(self, cosmoslics_pds: List[PersistenceDiagram], slics_pds: List[PersistenceDiagram], 
			  pixel_distinguishing_power: List[PixelDistinguishingPowerMap], min_count: int=0):
		self.pixel_distinguishing_power = pixel_distinguishing_power
		self.min_count = min_count
		super().__init__(cosmoslics_pds, slics_pds)

	def _build_training_set(self, pds: List[PersistenceDiagram]):
		# Build training set with betti number peaks
		dim_best_pixels = {}
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
						if perdi.betti_numbers_grids[dim]._transform_map()[x, y] >= self.min_count / len(perdi.dimension_pairs[dim]):
							passes = True
							break				

					if passes:
						x_ind.append(x)
						y_ind.append(y)

			dim_best_pixels[dim] = {
				'x_ind': x_ind,
				'y_ind': y_ind,
				'scatters': scatters
			}

		for cpd in pds:
			training_set_peaks['input'].append([val for key, val in cpd.cosm_parameters.items() if key != 'id'])
			for dim in [0, 1]:
				training_set_peaks[f'target_{dim}'].append(
					np.concatenate((
						[len(cpd.dimension_pairs[dim])],  # Number of features of dim
						cpd.betti_numbers_grids[dim]._transform_map()[dim_best_pixels[dim]['x_ind'], dim_best_pixels[dim]['y_ind']]
					))
				)

		# Merge dimensions in target data
		training_set_peaks['target'] = np.array([np.concatenate((t0, t1)) for t0, t1 in zip(training_set_peaks['target_0'], training_set_peaks['target_1'])])
		training_set_peaks['input'] = np.array(training_set_peaks['input'])

		return training_set_peaks
	

def find_local_maxima(rmap: PixelDistinguishingPowerMap, size=5):
	image = rmap.map
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