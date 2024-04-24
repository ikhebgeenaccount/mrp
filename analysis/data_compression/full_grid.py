from typing import List

import numpy as np

from analysis.cosmology_data import CosmologyData
from analysis.data_compression.index_compressor import IndexCompressor
from analysis.persistence_diagram import PersistenceDiagram


class FullGrid(IndexCompressor):

	def __init__(self, cosmoslics_datas: List[CosmologyData], slics_data: List[CosmologyData]):
		# Build set of indices that contain all entries
		indices = np.indices((len(slics_data[0].zbins_pds) * 2, 100, 100))
		super().__init__(cosmoslics_datas, slics_data, indices=indices.T)