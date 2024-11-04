

import os
import numpy as np

from analysis.persistence_diagram import BettiNumbersGridVarianceMap, BettiNumbersGrid


class CosmologyData:

	def __init__(self, 
			cosmology,
			zbins_pds=None,
			load_averages=False,  # TODO
			products_dir='products',
			n_cosmoslics_los=50
		):
		self.cosmology = cosmology
		self.cosm_parameters = list(zbins_pds.values())[0][0].cosm_parameters
		self.n_cosmoslics_los = n_cosmoslics_los

		if zbins_pds is not None:
			self.pds_count = len(list(zbins_pds.values())[0])
			self.zbins_pds = zbins_pds
			self.zbins_bngs = {
				zbin: {
					dim: [pd.betti_numbers_grids[dim] for pd in self.zbins_pds[zbin]] for dim in [0,1]
				} for zbin in self.zbins_pds
			}
			self.zbins_dimension_pairs_counts = {
				zbin: [pd.dimension_pairs_count for pd in self.zbins_pds[zbin]] 
				for zbin in self.zbins_pds
			}
		
		# Sort alphabetically
		sorted = np.sort(list(zbins_pds.keys()))
		# Sort by length of name to ensure crossbins at the end
		sorted = sorted[np.argsort([len(zb) for zb in sorted])]
		# Put widest bin in front
		sorted[0], sorted[1] = sorted[1], sorted[0]
		self.zbins = sorted

		self.products_dir = products_dir
		self.products_loc = os.path.join(products_dir, cosmology)

		self.calculate_averages()

	def calculate_averages(self, return_std=False):
		# Calculate average BNG for each zbin
		self.zbins_bngs_avg = {
			zbin: [
				BettiNumbersGrid(
					np.mean([pd.betti_numbers_grids[dim].map for pd in self.zbins_pds[zbin]], axis=0),
					self.zbins_pds[zbin][0].betti_numbers_grids[dim].x_range,
					self.zbins_pds[zbin][0].betti_numbers_grids[dim].y_range,
					dim
				) for dim in [0,1]
			] for zbin in self.zbins_pds
		}

		# Calculate std of BNG within each zbin
		self.zbins_bngs_std = {}

		for zbin in self.zbins_pds:

			self.zbins_bngs_std[zbin] = []

			for dim in [0, 1]:
				self.zbins_bngs_std[zbin].append(BettiNumbersGridVarianceMap(self.zbins_bngs[zbin][dim]))

				# # SLICS variance goes down as 1/sqrt(n_los_cosmoslics) (basically, number of measurements)
				# if self.cosmology == 'SLICS':
				# 	self.zbins_bngs_std[zbin][dim].map = self.zbins_bngs_std[zbin][dim].map / np.sqrt(self.n_cosmoslics_los)

		# Calculate average dimension pairs counts in each zbin
		self.dimension_pairs_count_avg = {
			zbin: [
				np.mean([pd.dimension_pairs_count[dim] for pd in self.zbins_pds[zbin]]) for dim in [0,1]
			] for zbin in self.zbins_pds
		}

		if return_std:
			return self.zbins_bngs_avg, self.zbins_bngs_std
		else:
			return self.zbins_bngs_avg
		
	def save(self):
		pass

	def load(self, path):
		pass