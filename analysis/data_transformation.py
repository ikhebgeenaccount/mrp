import numpy as np
import pandas as pd
import seaborn as sns

import os

from analysis.persistence_diagram import PersistenceDiagram

# Try PersistenceDiagram transformation

def transform_subtract(old_pd, new_pd):
	new_pd.dimension_pairs = {
		dim: np.array([old_pd.dimension_pairs[dim][:, 0], old_pd.dimension_pairs[dim][:, 1] - old_pd.dimension_pairs[dim][:, 0]]).T
		for dim in [0, 1]
	}

def transform_square(old_pd, new_pd):
	new_pd.dimension_pairs = {
		dim: np.array([old_pd.dimension_pairs[dim][:, 0], np.square(old_pd.dimension_pairs[dim][:, 1])]).T
		for dim in [0, 1]
	}

def transform_subtract_both(old_pd, new_pd):
	new_pd.dimension_pairs = {
		dim: np.array([old_pd.dimension_pairs[dim][:, 0] - old_pd.dimension_pairs[dim][:, 1], old_pd.dimension_pairs[dim][:, 1] - old_pd.dimension_pairs[dim][:, 0]]).T
		for dim in [0, 1]
	}
	
def rotation_matrix(angle):
	c = np.cos(np.radians(angle))
	s = np.sin(np.radians(angle))
	return np.array([[c, -s], [s, c]])

def transform_rotate(old_pd, new_pd):
	new_pd.dimension_pairs = {
		dim: np.einsum('ij,kj->ki', rotation_matrix(-45), old_pd.dimension_pairs[dim]) for dim in [0, 1]
	}

def try_transforms_and_plot(pds):
	for perdi in pds:

		perdi.plot(close=False)

		for dim in [0, 1]:
			data = pd.DataFrame(data=perdi.dimension_pairs[dim], columns=['birth', 'death'])
			gr = sns.jointplot(data=data, x='birth', y='death')

			gr.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)

		for transform, name in zip(
			[transform_subtract, transform_rotate],
			['death-birth', 'rotate']
		):
		
			new_pd = PersistenceDiagram(perdi.maps, cosmology=f'{perdi.cosmology}_transformed_{name}')

			transform(perdi, new_pd)
			fig, ax = new_pd.plot(close=False)
			# ax.set_ylim(ymin=-0.005, ymax=0.06)
			ax.set_ylabel(name)

			fig.savefig(os.path.join('plots', 'persistence_diagrams', f'{new_pd.cosmology}.pdf'))

			# Make Jointplot for dim 0, 1 separately
			for dim in [0, 1]:
				data = pd.DataFrame(data=new_pd.dimension_pairs[dim], columns=['birth', name])
				gr = sns.jointplot(data=data, x='birth', y=name)

				gr.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)

				# TODO: fit skewed Gaussian to rotated data

		break

def rotate_perdis(perdis):
	new_perdis = []
	for perdi in perdis:
		new_perdi = PersistenceDiagram(perdi.maps, cosmology=f'{perdi.cosmology}_rotated')

		transform_rotate(perdi, new_perdi)

		new_perdis.append(new_perdi)
	
	return new_perdis
