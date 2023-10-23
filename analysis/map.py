import os

import gudhi
import matplotlib.pyplot as plt
import numpy as np

from utils import file_system

def to_perseus_format(map, mask=None):
	# Ref: https://gudhi.inria.fr/python/latest/fileformats.html#file-formats

	output = '2\n'  # 2 dimensions
	# Add shape lines
	# First line is x length
	output += str(map.shape[1]) + '\n'
	# Second line is y length
	output += str(map.shape[0]) + '\n'

	masked_map = np.ma.array(map, mask=mask)

	# Add values from bottom left to top right
	for value in masked_map[::-1, :].flatten():
		if value is np.ma.masked:
			output += '\n'
		else:
			output += str(value) + '\n'

	return output


class Map:

	def __init__(self, filename):
		self.filename = filename
		self.filename_without_folder = filename.split('/')[-1]
		self.cosmology = filename.split('/')[-2]
		self._load()
		self._find_mask()
		self._apply_mask_set_inf()

	def __getitem__(self, item):
		return self.map[item]

	def _load(self):
		self.map = np.load(self.filename)

	def _find_mask(self):
		self.mask = self.map != 0

	def _apply_mask_set_inf(self):
		self.map[~self.mask] = np.inf

	def to_perseus_format(self):
		return to_perseus_format(self.map, self.mask)

	def plot(self):
		fig, ax = plt.subplots()
		imax = ax.imshow(self.map)
		fig.colorbar(imax)

	def _to_cubical_complex(self):
		# import tempfile
		# with tempfile.TemporaryFile() as file:
		# 	file.write(self.to_perseus_format().encode())
		# self.cubical_complex = gudhi.CubicalComplex(perseus_file=self.to_perseus_format())
		self.cubical_complex = gudhi.CubicalComplex(top_dimensional_cells=self.map)
		return self.cubical_complex
	
	def get_persistence(self):
		if hasattr(self, 'persistence'):
			return self.persistence

		if not hasattr(self, 'cubical_complex'):
			self._to_cubical_complex()
		self.persistence = self.cubical_complex.persistence()
		self._filter_persistence()
		self._separate_persistence_dimensions()
		return self.persistence
	
	def _filter_persistence(self):
		pass
	
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

			gaussian_size_in_sigma = 3
			sigma_in_pixel = scale_parameter / pixel_scale
			gaussian_size_in_pixel = sigma_in_pixel * 2 * gaussian_size_in_sigma  # Two times because symmetric Gaussian with both sides

			gaussian_kernel1d = gaussian(np.round(gaussian_size_in_pixel), std=sigma_in_pixel)
			gaussian_kernel = np.outer(gaussian_kernel1d, gaussian_kernel1d)

			heatmap = convolve2d(hist, gaussian_kernel, mode='same')
		
			self.heatmaps.append(Heatmap(heatmap, data_range, data_range, dimension))

			self.heatmaps[-1].save(os.path.join('heatmaps', self.cosmology, self.filename_without_folder))

			self.heatmaps[-1].save_figure(os.path.join('plots', 'heatmaps', self.cosmology, self.filename_without_folder))


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
