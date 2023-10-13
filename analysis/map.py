import os

import gudhi
import matplotlib.pyplot as plt
import numpy as np

from utils import file_system

class Map:

	def __init__(self, filename):
		self.filename = filename
		self.filename_without_folder = filename.split('/')[-1]
		self.cosmology = filename.split('/')[-2]
		self._load()

	def __getitem__(self, item):
		return self.map[item]

	def _load(self):
		self.map = np.load(self.filename)

	def plot(self):
		fig, ax = plt.subplots()
		imax = ax.imshow(self.map)
		fig.colorbar(imax)

	def _to_cubical_complex(self):
		self.cubical_complex = gudhi.CubicalComplex(top_dimensional_cells=self.map)
		return self.cubical_complex
	
	def get_persistence(self):
		if hasattr(self, 'persistence'):
			return self.persistence

		if not hasattr(self, 'cubical_complex'):
			self._to_cubical_complex()
		self.persistence = self.cubical_complex.persistence()
		self._separate_persistence_dimensions()
		return self.persistence
	
	def _separate_persistence_dimensions(self):
		# Persistence is list of (dimension, (birth, death))

		# Create lists of (birth, death) pairs for each dimension
		dimension_pairs = {
			'all': []
		}
		for (dimension, pair) in self.persistence:
			if dimension not in dimension_pairs:
				dimension_pairs[dimension] = [pair]
			else:
				dimension_pairs[dimension].append(pair)
			
			dimension_pairs['all'].append(pair)
		
		self.dimension_pairs = {dimension: np.array(dimension_pairs[dimension]) for dimension in dimension_pairs}
	
	def get_betti_numbers(self):
		self.get_persistence()
		self.betti_numbers = self.cubical_complex.betti_numbers()
		return self.betti_numbers
	
	def get_persistent_betti_numbers(self, birth_before, death_after):
		"""
		Compute the persistent Betti Numbers of the map 
		"""
		self.get_persistence()
		self.persistent_betti_numbers = self.cubical_complex.persistent_betti_numbers(from_value=birth_before, to_value=death_after)
		return self.persistent_betti_numbers

	def generate_heatmaps(self, resolution=1000):
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

		def get_range(axis):
			# Filter the data array for finite numbers such that our boundaries are properly defined
			data_arr = self.dimension_pairs['all'][:,axis][np.isfinite(self.dimension_pairs['all'][:,axis])]
			return np.linspace(np.min(data_arr), np.max(data_arr), resolution)

		# The 2D map of the scatter points still has the same min and max in x and y
		for dimension in self.dimension_pairs:
			if dimension == 'all':
				continue

			birth_range = get_range(axis=0)
			death_range = get_range(axis=1)

			# We see these range values as bin edges (left side) for the heatmap pixels
			heatmap = np.zeros((resolution, resolution))

			for pair in self.dimension_pairs[dimension]:
				if not np.all(np.isfinite(pair)):
					continue
				birth_coord = np.sum(birth_range >= pair[0]) - 1
				death_coord = np.sum(death_range >= pair[1]) - 1

				heatmap[death_coord, birth_coord] = 1.

			# Scale parameter of the Gaussian
			# Set to 1/25th of the range, similar value as Heydenreich+2022
			pixel_scale = np.abs(birth_range[0] - birth_range[1])  # FIXME: pixels are not perfect squares, as death and birth ranges are different
			# Determine the scale parameter (std, sigma) of the Gaussian kernel
			scale_parameter = 1. / 25. * np.abs(birth_range[0] - birth_range[-1])

			gaussian_size_in_sigma = 3
			sigma_in_pixel = scale_parameter / pixel_scale
			gaussian_size_in_pixel = sigma_in_pixel * 2 * gaussian_size_in_sigma  # Two times because symmetric Gaussian with both sides

			gaussian_kernel1d = gaussian(np.round(gaussian_size_in_pixel), std=sigma_in_pixel)
			gaussian_kernel = np.outer(gaussian_kernel1d, gaussian_kernel1d)

			heatmap = convolve2d(heatmap, gaussian_kernel, mode='same')
		
			self.heatmaps.append(Heatmap(heatmap, (birth_range[0], birth_range[-1]), (death_range[0], death_range[-1]), dimension))

			file_system.check_folder_exists(os.path.join('heatmaps', self.cosmology))
			self.heatmaps[-1].save(os.path.join('heatmaps', self.cosmology, self.filename_without_folder))

		# Diagnostic plots
		# import os
		# fig, ax = plt.subplots()
		# ax.plot(gaussian_kernel1d)
		# fig.savefig(os.path.join('plots', 'gaussian_kernel_1d.png'))
		
		# fig, ax = plt.subplots()
		# ax.imshow(gaussian_kernel)
		# fig.savefig(os.path.join('plots', 'gaussian_kernel_2d.png'))

	
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