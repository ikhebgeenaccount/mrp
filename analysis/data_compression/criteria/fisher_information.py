

import numpy as np
from analysis.data_compression.compressor import Compressor
from analysis.data_compression.criteria.criterium import Criterium
from analysis.data_compression.full_grid import FullGrid


class FisherInformation(Criterium):

	def __init__(self, cosmoslics_datas, slics_data, fisher_info_increase):
		self.cosmoslics_datas = cosmoslics_datas
		self.slics_data = slics_data

		self.fisher_info_increase = fisher_info_increase
		self.prev_fisher_info = 0.
		self.fisher_info_vals = []

	def acceptance_func(self, compressor: Compressor):
		new_fisher_info = self.criterium_value(compressor)

		if new_fisher_info / self.prev_fisher_info >= 1. + self.fisher_info_increase:
			self.prev_fisher_info = new_fisher_info

			self.fisher_info_vals.append(new_fisher_info)
			return True
		return False

	def criterium_value(self, compressor: Compressor):
		return compressor.fisher_matrix[1, 1]

	def pixel_scores(self):
		full_grid = FullGrid(self.cosmoslics_datas, self.slics_data)

		# Sort full_grid by fisher info
		# Minus to make argsort descending order
		return np.reshape(np.max(-full_grid.fisher_matrix_per_entry, axis=(0, 1)), (len(self.slics_data[0].zbins_pds), 2, 100, 100))