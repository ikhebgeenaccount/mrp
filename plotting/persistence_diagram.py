import os
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np

from analysis.map import Map

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