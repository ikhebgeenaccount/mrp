from typing import List

import numpy as np
from analysis.data_compression.compressor import Compressor
from analysis.persistence_diagram import PersistenceDiagram

class NumberOfFeaturesCompressor(Compressor):

	def __init__(self, cosmoslics_pds: List[PersistenceDiagram], slics_pds: List[PersistenceDiagram]):
		super().__init__(cosmoslics_pds, slics_pds)

	def _build_training_set(self, pds: List[PersistenceDiagram]):
		# Build training set with betti number peaks
		dim_best_pixels = {}
		training_set_peaks = {
			'name': 'number_of_features',
			'input': [],
			'target_0': [],
			'target_1': []
		}

		for cpd in pds:
			training_set_peaks['input'].append([val for key, val in cpd.cosm_parameters.items() if key != 'id'])
			
			for dim in [0, 1]:
				training_set_peaks[f'target_{dim}'].append(
					[len(cpd.dimension_pairs[dim])]
				)

		# Merge dimensions in target data
		training_set_peaks['target'] = np.array([np.concatenate((t0, t1)) for t0, t1 in zip(training_set_peaks['target_0'], training_set_peaks['target_1'])])
		training_set_peaks['input'] = np.array(training_set_peaks['input'])

		return training_set_peaks