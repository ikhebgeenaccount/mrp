import os
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np

from analysis.map import Map
import analysis.cosmologies as cosmologies
from utils import file_system

class PersistenceDiagram:

	def __init__(self, maps: List[Map], cosmology=None):
		if len(maps) > 0:
			self.dimension_pairs = maps[0].dimension_pairs.copy()

			self.maps = maps

			for map in maps[1:]:
				for dimension in self.dimension_pairs:
					self.dimension_pairs[dimension] = np.append(self.dimension_pairs[dimension], map.dimension_pairs[dimension], axis=0)

			for dim in self.dimension_pairs:
				# np.min collapses the isfinite check to cover the pair of values instead of only one coordinate
				self.dimension_pairs[dim] = self.dimension_pairs[dim][np.min(np.isfinite(self.dimension_pairs[dim]), axis=1)]

		if cosmology is None:
			self.cosmology = maps[0].cosmology
			self.cosmology_id = maps[0].cosmology_id
		else:
			self.cosmology = cosmology
			self.cosmology_id = None

		self.cosm_parameters = cosmologies.get_cosmological_parameters(self.cosmology).to_dict()

		if len(maps) == 1:
			# Save the line of sight discriminator if we only have one map
			self.los = maps[0].filename_without_folder

	def plot(self, close=True):
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
		ax.set_title(self.cosmology)
		lim = 0.06
		ax.set_ylim(ymin=-lim, ymax=lim)
		ax.set_xlim(xmin=-lim, xmax=lim)

		eq_line = np.linspace(-lim, lim, 2)
		ax.plot(eq_line, eq_line, linestyle='--', color='grey')

		fig.savefig(os.path.join('plots', 'persistence_diagrams', f'{self.cosmology}.png'))

		if close:
			plt.close(fig)
			return None
		else:
			return fig, ax

	def add_average_lines(self):
		# Average death and birth
		self.ax.axhline(y=np.average(self.dimension_pairs['all'][:, 1][np.isfinite(self.dimension_pairs['all'][:, 1])]), linestyle='--', color='black')
		self.ax.axvline(x=np.average(self.dimension_pairs['all'][:, 0][np.isfinite(self.dimension_pairs['all'][:, 0])]), linestyle='--', color='black')
		
		# Average of map
		all_maps_avg = np.average([np.average(map.map) for map in self.maps])
		self.ax.axvline(x=all_maps_avg, linestyle='--', color='grey')
		self.ax.axhline(y=all_maps_avg, linestyle='--', color='grey')

	def get_persistent_betti_numbers(self, birth_before: np.ndarray, death_after: np.ndarray, dimension):

		# Count the number of scatter points that have birth < birth_before and death > death_after
		pairs = self.dimension_pairs[dimension]

		number_of_features = pairs.shape[0]
		number_of_test_coords = birth_before.shape[0]

		birth_check = np.less(
			np.broadcast_to(pairs[:, 0], (number_of_test_coords, number_of_features)), 
			birth_before.reshape((-1, 1))
		)
		death_check = np.greater(
			np.broadcast_to(pairs[:, 1], (number_of_test_coords, number_of_features)), 
			death_after.reshape((-1, 1))
		)

		# Broadcast one array from [[a, b, c], [d, e, f]] to [[[a, b, c], [d, e, f]], [[a, b, c], [d, e, f]]]
		# The other from [[a, b, c], [d, e, f]] to [[[a, b, c], [a, b, c]], [[d, e, f], [d, e, f]]]
		# to be able to multiply and find every combination of born before and died after check
		birth_side = np.broadcast_to(birth_check, (number_of_test_coords, number_of_test_coords, number_of_features))
		death_side = np.repeat(death_check, number_of_test_coords, axis=0).reshape(number_of_test_coords, number_of_test_coords, number_of_features)

		return np.sum(birth_side * death_side, axis=2)
	
	def generate_betti_numbers_grids(self, resolution=100, regenerate=False, data_ranges_dim=None):
		
		self.betti_numbers_grids = {}

		if not regenerate and os.path.exists(os.path.join('products', 'betti_number_grids', self.cosmology)):
			self.betti_numbers_grids[0] = load_betti_numbers_grid(os.path.join('products', 'betti_number_grids', self.cosmology), 0)
			self.betti_numbers_grids[1] = load_betti_numbers_grid(os.path.join('products', 'betti_number_grids', self.cosmology), 1)
			return self.betti_numbers_grids

		for dimension in self.dimension_pairs:
			if dimension == 'all':
				continue

			data_range = [
				np.min(self.dimension_pairs[dimension]), 
				np.max(self.dimension_pairs[dimension])
			]

			if data_ranges_dim is None:
				birth_linspace = np.linspace(*data_range, resolution)
				death_linspace = np.linspace(*data_range, resolution)
			else:
				birth_linspace = np.linspace(*data_ranges_dim[dimension], resolution)
				death_linspace = np.linspace(*data_ranges_dim[dimension], resolution)

			betti_numbers_grid = np.zeros((resolution, resolution))

			betti_numbers_grid = self.get_persistent_betti_numbers(birth_linspace, death_linspace, dimension)
			# Normalize
			betti_numbers_grid = betti_numbers_grid / np.max(betti_numbers_grid)
			
			self.betti_numbers_grids[dimension] = BettiNumbersGrid(betti_numbers_grid, 
				[birth_linspace[0], birth_linspace[-1]], 
				[death_linspace[0], death_linspace[-1]], 
				dimension=dimension
			)
			
			if hasattr(self, 'los'):
				self.betti_numbers_grids[dimension].save(os.path.join('products', 'betti_numbers_grid', self.cosmology, self.los))

				self.betti_numbers_grids[dimension].save_figure(
					os.path.join('plots', 'betti_number_grids', self.cosmology, self.los), 
					scatter_points=(self.dimension_pairs[dimension][:, 0], self.dimension_pairs[dimension][:, 1])
				)

			else:
				self.betti_numbers_grids[dimension].save(os.path.join('products', 'betti_numbers_grid', self.cosmology))

				self.betti_numbers_grids[dimension].save_figure(
					os.path.join('plots', 'betti_number_grids', self.cosmology), 
					scatter_points=(self.dimension_pairs[dimension][:, 0], self.dimension_pairs[dimension][:, 1])
				)
		
		return self.betti_numbers_grids


	def generate_heatmaps(self, resolution=1000, gaussian_kernel_size_in_sigma=3, regenerate=False):
		"""
		Generates the heatmaps corresponding to the birth and death times given calculated by get_persistence.
		The heatmap is a convolution of the birth and death times (scatterplot) with a 2D Gaussian.
		:param resolution: The number of pixels in one axis, the resulting heatmap is always a square of size resolution x resolution
		"""
		from scipy.signal import convolve2d
		from scipy.signal.windows import gaussian

		self.heatmaps = {}

		if not regenerate and os.path.exists(os.path.join('products', 'heatmaps', self.cosmology)):
			self.heatmaps[0] = load_heatmap(os.path.join('products', 'heatmaps', self.cosmology), 0)
			self.heatmaps[1] = load_heatmap(os.path.join('products', 'heatmaps', self.cosmology), 1)
			return self.heatmaps

		# convolve2d takes two arrays as input
		# We need to generate a 2D array with 1s in the spot of each (birth, death) scatter point
		# which will be convolved with a 2D Gaussian

		# The 2D map of the scatter points still has the same min and max in x and y
		for dimension in self.dimension_pairs:
			if dimension == 'all':
				continue

			x = self.dimension_pairs[dimension][:, 0]
			y = self.dimension_pairs[dimension][:, 1]

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
		
			self.heatmaps[dimension] = Heatmap(heatmap, data_range, data_range, dimension)

			if hasattr(self, 'los'):
				self.heatmaps[dimension].save(os.path.join('products', 'heatmaps', self.cosmology, self.los))

				self.heatmaps[dimension].save_figure(os.path.join('plots', 'heatmaps', self.cosmology, self.los))#, scatter_points=(x, y))
			else:
				self.heatmaps[dimension].save(os.path.join('products', 'heatmaps', self.cosmology))

				self.heatmaps[dimension].save_figure(os.path.join('plots', 'heatmaps', self.cosmology))#, scatter_points=(x, y))
		
		return self.heatmaps
	
	def save_fig(self, figure, base_path, file_name):
		if hasattr(self, 'los'):
			figure.savefig(os.path.join(base_path, self.los, file_name))
		else:
			figure.savefig(os.path.join(base_path, file_name))


def load_heatmap(path, dimension):
	"""
	Loads a saved Heatmap from path.
	If directory structure is as follows:
	/data/heatmaps/hm1/
				heatmap_0.py
				heatmap_1.py
				birth_range0.py
				birth_range1.py
				death_range0.py
				death_range1.py
	Then the heatmap of dimension 0 can be loaded through
		load_heatmap('/data/heatmaps/hm1', 0)
	or dimension 1
		load_heatmap('/data/heatmaps/hm1', 1)
	"""
	return _load_ranged_map(path, dimension, Heatmap)


def load_betti_numbers_grid(path, dimension):
	return _load_ranged_map(path, dimension, BettiNumbersGrid)


def load_betti_numbers_variance_map(path, dimension):
	return _load_ranged_map(path, dimension, BettiNumbersGridVarianceMap)


def _load_ranged_map(path, dimension, map_type):
	rmap = map_type(None, None, None, dimension)
	rmap.load(path)
	return rmap

	
class BaseRangedMap:

	def __init__(self, map, x_range, y_range, dimension, name):
		self.name = name
		self.map = map
		self.x_range = x_range
		self.y_range = y_range
		self.dimension = dimension
	
	def save(self, path):
		file_system.check_folder_exists(path)
		np.save(os.path.join(path, f'{self.name}_{self.dimension}.npy'), self.map)
		np.save(os.path.join(path, f'x_range_{self.dimension}.npy'), self.x_range)
		np.save(os.path.join(path, f'y_range_{self.dimension}.npy'), self.y_range)

	def load(self, path):
		self.map = np.load(os.path.join(path, f'{self.name}_{self.dimension}.npy'))
		self.x_range = np.load(os.path.join(path, f'x_range_{self.dimension}.npy'))
		self.y_range = np.load(os.path.join(path, f'y_range_{self.dimension}.npy'))

	def get_axis_values(self, axis):
		if axis == 'x':
			return np.linspace(self.x_range[0], self.x_range[1], num=len(self.map[0]))
		elif axis == 'y':
			return np.linspace(self.y_range[0], self.y_range[1], num=len(self.map))

	def save_figure(self, path, scatter_points=None, title=None, save_name=None):
		file_system.check_folder_exists(path)
		fig, ax = plt.subplots()
		imax = ax.imshow(self._transform_map(), aspect='equal', extent=(*self.x_range, *self.y_range))
		fig.colorbar(imax)

		if scatter_points is not None:
			ax.scatter(*scatter_points, s=3, alpha=.6, color='red')

		if title is not None:
			ax.set_title(title)

		if save_name is None:
			fig.savefig(os.path.join(path, f'{self.name}_{self.dimension}.png'))
		else:
			fig.savefig(os.path.join(path, f'{self.name}_{self.dimension}_{save_name}.png'))
		plt.close(fig)

	def _transform_map(self):
		return self.map
	

class Heatmap(BaseRangedMap):

	def __init__(self, heatmap, birth_range, death_range, dimension):
		super().__init__(heatmap, birth_range, death_range, dimension, name='heatmap')

	def _transform_map(self):
		return self.map.T[::-1,:]


class BettiNumbersGrid(BaseRangedMap):

	def __init__(self, betti_numbers_grid, birth_range, death_range, dimension):
		super().__init__(betti_numbers_grid, birth_range, death_range, dimension, name='betti_numbers_grid')

	def _transform_map(self):
		return self.map[::-1, :]
	

class BettiNumbersGridVarianceMap(BaseRangedMap):

	def __init__(self, betti_numbers_grids: List[BettiNumbersGrid], birth_range, death_range, dimension):
		super().__init__(None, birth_range, death_range, dimension, name='betti_numbers_grid_variance_map')

		grids = []
		for bng in betti_numbers_grids:
			grids.append(bng.map)

		self.map = np.std(grids, axis=0)

	def _transform_map(self):
		return self.map[::-1, :]


class PixelDistinguishingPowerMap(BaseRangedMap):

	def __init__(self, cosmoslics_bngs: List[BettiNumbersGrid], slics_bng: BettiNumbersGrid, slics_variance: BettiNumbersGridVarianceMap, dimension):
		# Allow for an empty map to be created in which the data is loaded later
		if cosmoslics_bngs is None and slics_bng is None and slics_variance is None:
			super().__init__(None, None, None, dimension, 'pixel_distinguishing_power')
		# If arrays are passed, check that the ranges are the same
		if not np.all([np.allclose(cbng.y_range, slics_bng.y_range) for cbng in cosmoslics_bngs]):
			raise ValueError('Death ranges are different for cosmoSLICS and SLICS BettiNumbersGrids')
		if not np.all([np.allclose(cbng.x_range, slics_bng.x_range) for cbng in cosmoslics_bngs]):
			raise ValueError('Birth ranges are different for cosmoSLICS and SLICS BettiNumbersGrids')
		if not np.all([np.allclose(slics_bng.x_range, slics_variance.x_range), np.allclose(slics_bng.y_range, slics_variance.y_range)]):
			raise ValueError('Ranges of SLICS BettiNumbersGrid and BettiNumbersVarianceGrid are different')
		super().__init__(None, slics_bng.x_range, slics_bng.y_range, dimension, 'pixel_distinguishing_power')

		self.map = np.mean(np.square([cbng.map for cbng in cosmoslics_bngs] - slics_bng.map) / slics_variance.map, axis=0)

	def _transform_map(self):
		return self.map[::-1, :]