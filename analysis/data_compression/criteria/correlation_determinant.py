

import numpy as np
from analysis.data_compression.compressor import Compressor
from analysis.data_compression.criteria.criterium import Criterium


class CorrelationDeterminant(Criterium):

	def __init__(self, max_fraction_decrease=0.9):
		self.det_values = []
		self.max_fraction_decrease = max_fraction_decrease

	def acceptance_func(self, compressor: Compressor):
		new_det = self.criterium_value(compressor)
		# If there are no accepted values yet, the first one is always accepted
		if len(self.det_values) == 0:
			self.det_values.append(new_det)
			return True

		if new_det / self.det_values[-1] > 1. - self.max_fraction_decrease:
			self.det_values.append(new_det)
			return True
		return False
		
	def criterium_value(self, compressor: Compressor):
		if not hasattr(compressor, 'slics_crosscorr_matrix'):
			compressor._build_crosscorr_matrix()
		return np.linalg.det(compressor.slics_crosscorr_matrix)