from typing import List

import numpy as np

from analysis.data_compression.index_compressor import IndexCompressor
from analysis.persistence_diagram import PersistenceDiagram


class FullGrid(IndexCompressor):

	def __init__(self, cosmoslics_pds: List[PersistenceDiagram], slics_pds: List[PersistenceDiagram]):
		# Build set of indices that contain all entries
		indices = np.indices((2, 100, 100))
		super().__init__(cosmoslics_pds, slics_pds, indices=indices.T)