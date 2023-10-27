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
