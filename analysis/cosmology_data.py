

import os
import numpy as np

from analysis.persistence_diagram import BettiNumbersGridVarianceMap


class CosmologyData:

	def __init__(self, 
			cosmology,
			zbins_pds=None,
			load_averages=False,  # TODO
			products_dir='products'
		):
		self.cosmology = cosmology
		self.cosm_parameters = zbins_pds.values()[0].cosm_parameters

		if zbins_pds is not None:
			self.zbins_pds = zbins_pds
			self.zbins_bngs = {
				zbin: {
					dim: [pd.betti_numbers_grids[dim] for pd in self.zbins_pds[zbin]] for dim in [0,1]
				} for zbin in self.zbins_pds
			}

		self.products_dir = products_dir
		self.products_loc = os.path.join(products_dir, cosmology)

		self.calculate_average_bngs()

	def calculate_average_bngs(self, return_std=False):
		self.zbins_bngs_avg = {
			zbin: [
				np.mean([pd.betti_numbers_grids[dim].map for pd in self.zbins_pds[zbin]], axis=0) for dim in [0,1]
			] for zbin in self.zbins_pds
		}
		self.zbins_bngs_std = {
			zbin: {
				BettiNumbersGridVarianceMap(self.zbins_bngs[zbin][dim]) for dim in [0,1]
			} for zbin in self.zbins_pds
		}

		self.dimension_pairs_count_avg = {
			zbin: [
				np.mean([pd.dimension_pairs_count[dim] for pd in self.zbins_pds[zbin]]) for dim in [0,1]
			] for zbin in self.zbins_pds
		}

		if return_std:
			return self.zbins_pds_avg, self.zbins_pds_std
		else:
			return self.zbins_pds_avg