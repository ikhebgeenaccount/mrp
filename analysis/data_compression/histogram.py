
from typing import List
from analysis.data_compression.compressor import Compressor
from analysis.persistence_diagram import PersistenceDiagram
from analysis.data_transformation import rotate_perdis

import numpy as np

class HistogramCompressor(Compressor):

	def __init__(self, cosmoslics_pds: List[PersistenceDiagram], slics_pds: List[PersistenceDiagram]):
		super().__init__(cosmoslics_pds, slics_pds)

	def _build_training_set(self, pds: List[PersistenceDiagram]):
		# Build training set based on rotated Persistence Diagrams Histograms
		cosmoslics_pds_rot = rotate_perdis(pds)
		training_set_histboth = {
			'name': 'histogram_both',
			'input': [],
			'target': []
		}
		training_set_histx = {
			'name': 'histogram_x',
			'input': [],
			'target': []
		}

		for perdi in cosmoslics_pds_rot:
			training_set_histboth['input'].append([val for key, val in perdi.cosm_parameters.items() if key != 'id'])
			training_set_histx['input'].append([val for key, val in perdi.cosm_parameters.items() if key != 'id'])

			curr_targ_both = []
			curr_targ_x = []

			for dim in [0, 1]:
				histx, _ = np.histogram(perdi.dimension_pairs[dim][:, 0], bins=20, density=False)
				histy, _ = np.histogram(perdi.dimension_pairs[dim][:, 1], bins=20, density=False)

				curr_targ_both.append(histx)
				curr_targ_both.append(histy)

				curr_targ_x.append(histx)

			training_set_histboth['target'].append(np.array(curr_targ_both).flatten())
			training_set_histx['target'].append(np.array(curr_targ_x).flatten())

		for t_set in [training_set_histboth, training_set_histx]:
			t_set['target'] = np.array(t_set['target'])

		return training_set_histboth