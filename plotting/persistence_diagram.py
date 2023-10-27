import os
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np

from analysis.map import Map
from utils import file_system

class PersistenceDiagram:

	def __init__(self, maps: List[Map]):
		self.dimension_pairs = maps[0].dimension_pairs.copy()

		self.maps = maps

		for map in maps[1:]:
			for dimension in self.dimension_pairs:
				self.dimension_pairs[dimension] = np.append(self.dimension_pairs[dimension], map.dimension_pairs[dimension], axis=0)

		# Scatter each dimension separately
		fig, ax = plt.subplots()
		ax.set_xlabel('Birth')
		ax.set_ylabel('Death')
		for dimension in self.dimension_pairs:
			# Turn into np array for easy slicing
			pairs = self.dimension_pairs[dimension]

			# ax.scatter(pairs[np.isfinite(np.linalg.norm(pairs, axis=1)), 0], pairs[np.isfinite(np.linalg.norm(pairs, axis=1)), 1], label=f'{dimension}', s=3)
			ax.scatter(pairs[:, 0], pairs[:, 1], label=f'{dimension}', s=3)
		
		ax.legend()
		ax.set_title(maps[0].cosmology)
		lim = 0.06
		ax.set_ylim(ymin=-lim, ymax=lim)
		ax.set_xlim(xmin=-lim, xmax=lim)

		# TODO: add diagonal

		self.ax = ax
		self.fig = fig
		self.cosmology = maps[0].cosmology

	def save(self, path):
		self.fig.savefig(os.path.join(path, f'{self.cosmology}.png'))

	def add_average_lines(self):
		# Average death and birth
		self.ax.axhline(y=np.average(self.dimension_pairs['all'][:, 1][np.isfinite(self.dimension_pairs['all'][:, 1])]), linestyle='--', color='black')
		self.ax.axvline(x=np.average(self.dimension_pairs['all'][:, 0][np.isfinite(self.dimension_pairs['all'][:, 0])]), linestyle='--', color='black')
		
		# Average of map
		all_maps_avg = np.average([np.average(map.map) for map in self.maps])
		self.ax.axvline(x=all_maps_avg, linestyle='--', color='grey')
		self.ax.axhline(y=all_maps_avg, linestyle='--', color='grey')

	def get_persistent_betti_numbers(self, birth_before, death_after):
		betti_numbers = []

		for dim in self.dimension_pairs:
			if dim == 'all':
				continue

			# Count the number of scatter points that have birth < birth_before and death > death_after
			pairs = self.dimension_pairs[dim]
			birth_check = pairs[:, 0] < birth_before
			death_check = pairs[:, 1] > death_after

			betti_dim = np.sum(birth_check * death_check)

			betti_numbers.append(betti_dim)
		
		return betti_numbers
	
	def compute_betti_numbers_grid(self, resolution=100):
		pass

	def generate_heatmaps(self, resolution=1000, gaussian_kernel_size_in_sigma=3):
		"""
		Generates the heatmaps corresponding to the birth and death times given calculated by get_persistence.
		The heatmap is a convolution of the birth and death times (scatterplot) with a 2D Gaussian.
		:param resolution: The number of pixels in one axis, the resulting heatmap is always a square of size resolution x resolution
		"""
		from scipy.signal import convolve2d
		from scipy.signal.windows import gaussian

		self.heatmaps = []

		# convolve2d takes two arrays as input
		# We need to generate a 2D array with 1s in the spot of each (birth, death) scatter point
		# which will be convolved with a 2D Gaussian

		# The 2D map of the scatter points still has the same min and max in x and y
		for dimension in self.dimension_pairs:
			if dimension == 'all':
				continue

			filter_for_finite = np.min(np.isfinite(self.dimension_pairs[dimension]), axis=1)

			x = self.dimension_pairs[dimension][:, 0][filter_for_finite]
			y = self.dimension_pairs[dimension][:, 1][filter_for_finite]

			# We want each pixel to be a square, so we need to find the largest range of values to cover
			data_range = [
				np.min(self.dimension_pairs[dimension][np.isfinite(self.dimension_pairs[dimension])]), 
				np.max(self.dimension_pairs[dimension][np.isfinite(self.dimension_pairs[dimension])])
			]

			# range = [x_range, y_range], set to equal so we have square pixels
			hist, bin_edges_x, bin_edges_y = np.histogram2d(x, y, bins=resolution, range=[data_range, data_range])

			# Scale parameter of the Gaussian
			pixel_scale = np.abs(data_range[1] - data_range[0]) / resolution
			# Determine the scale parameter (std, sigma) of the Gaussian kernel
			# Set to 1/25th of the range, similar value as Heydenreich+2022
			scale_parameter = 1. / 25. * np.abs(data_range[1] - data_range[0])

			sigma_in_pixel = scale_parameter / pixel_scale
			gaussian_size_in_pixel = sigma_in_pixel * 2 * gaussian_kernel_size_in_sigma  # Two times because symmetric Gaussian with both sides

			gaussian_kernel1d = gaussian(np.round(gaussian_size_in_pixel), std=sigma_in_pixel)
			gaussian_kernel = np.outer(gaussian_kernel1d, gaussian_kernel1d)

			heatmap = convolve2d(hist, gaussian_kernel, mode='same')
		
			self.heatmaps.append(Heatmap(heatmap, data_range, data_range, dimension))

			self.heatmaps[-1].save(os.path.join('heatmaps', self.cosmology))

			self.heatmaps[-1].save_figure(os.path.join('plots', 'heatmaps', self.cosmology))
		
		return self.heatmaps


def load_heatmap(path, dimension):
	"""
	Loads a saved Heatmap from path.
	If directory structure is as follows:
	/data/heatmaps/hm1/
				heatmap0.py
				heatmap1.py
				birth_range0.py
				birth_range1.py
				death_range0.py
				death_range1.py
	Then the heatmap of dimension 0 can be loaded through
		load_heatmap('/data/heatmaps/hm1', 0)
	or dimension 1
		load_heatmap('/data/heatmaps/hm1', 1)
	"""
	heatmap = Heatmap(None, None, None, dimension)
	heatmap.load(path)
	return heatmap

	
class Heatmap:

	def __init__(self, heatmap, birth_range, death_range, dimension):
		self.heatmap = heatmap
		self.birth_range = birth_range
		self.death_range = death_range
		self.dimension = dimension

	def __getitem__(self, item):
		return self.heatmap[item]
	
	def save(self, path):
		file_system.check_folder_exists(path)
		np.save(os.path.join(path, f'heatmap_{self.dimension}.npy'), self.heatmap)
		np.save(os.path.join(path, f'birth_range_{self.dimension}.npy'), self.birth_range)
		np.save(os.path.join(path, f'death_range_{self.dimension}.npy'), self.death_range)

	def load(self, path):
		self.heatmap = np.load(os.path.join(path, f'heatmap_{self.dimension}.npy'))
		self.birth_range = np.load(os.path.join(path, f'birth_range_{self.dimension}.npy'))
		self.death_range = np.load(os.path.join(path, f'death_range_{self.dimension}.npy'))

	def save_figure(self, path):
		file_system.check_folder_exists(path)
		fig, ax = plt.subplots()
		ax.imshow(self.heatmap[:,::-1], aspect='equal', extent=(*self.birth_range, *self.death_range))
		fig.savefig(os.path.join(path, f'heatmap_{self.dimension}.png'))
		plt.close(fig)