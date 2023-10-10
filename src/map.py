import gudhi
import matplotlib.pyplot as plt
import numpy as np

class Map:

	def __init__(self, filename):
		self.filename = filename
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
	
	def plot_persistence(self):
		# TODO: matplotlib not found when gudhi tries to plot
		# ax = gudhi.plot_persistence_diagram(self.get_persistence())
		# return ax
		
		# Scatter each dimension separately
		fig, ax = plt.subplots()
		ax.set_xlabel('Birth')
		ax.set_ylabel('Death')
		for dimension in self.dimension_pairs:
			# Turn into np array for easy slicing
			pairs = self.dimension_pairs[dimension]

			ax.scatter(pairs[:, 0], pairs[:, 1], label=f'{dimension}', s=3)
		
		ax.legend()

		return ax

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
			scale_parameter = 1. / 25. * np.abs(birth_range[0] - birth_range[-1])
			# TODO: how to decide on Gaussian parameters?
			scale_parameter = 10
			window_size = 100

			gaussian_kernel1d = gaussian(window_size, std=scale_parameter)
			gaussian_kernel = np.outer(gaussian_kernel1d, gaussian_kernel1d)

			heatmap = convolve2d(heatmap, gaussian_kernel, mode='same')
		
			self.heatmaps.append(Heatmap(heatmap, (birth_range[0], birth_range[-1]), (death_range[0], death_range[-1])))

	
class Heatmap:

	def __init__(self, heatmap, birth_range, death_range):
		self.heatmap = heatmap
		self.birth_range = birth_range
		self.death_range = death_range

	def __getitem__(self, item):
		return self.heatmap[item]