import os

import matplotlib.pyplot as plt
import pandas as pd
import treecorr

def build_treecorr_catalog(df, shear_cols=None, kappa_col=None):
	if shear_cols is None:
		shear_cols = ['eps_data1', 'eps_data2']

	return treecorr.Catalog(ra=df['RA'], dec=df['DEC'], g1=df[shear_cols[0]], g2=df[shear_cols[1]], w=df['w'], k=df[kappa_col] if kappa_col is not None else None,
							ra_units='deg', dec_units='deg')

def read_treecorr_result(result_file):
	return pd.read_csv(result_file, delim_whitespace=True, skiprows=2, 
				names=['r_nom', 'meanr', 'meanlogr', 'xip', 'xim', 'xip_im', 'xim_im', 'sigma_xip', 'sigma_xim', 'weight', 'npairs'])


def plot_correlation_function(df):
	formatting = {
		'fmt': 'o',
		'markersize': 3,
		'capsize': 3,
		'elinewidth': 1,
		
	}

	fig, ax = plt.subplots()
	ax.errorbar(df.r_nom, df.xip, label='$\\xi_+$', yerr=df.sigma_xip, **formatting)
	ax.errorbar(df.r_nom, df.xim, label='$\\xi_-$', yerr=df.sigma_xim, **formatting)
	# ax.hist(df_athena.xi_p, bins=df_athena.theta + (df_athena.theta[1] - df_athena.theta[0]) / 2, label='$\ksi_+$')
	# ax.hist(df_athena.xi_m, bins=df_athena.theta + (df_athena.theta[1] - df_athena.theta[0]) / 2, label='$\ksi_-$')
	ax.legend()
	ax.semilogx()	
	ax.axhline(xmin=0, xmax=1, y=0., color='black', linestyle='--')

	fig.savefig(os.path.join('plots', f'2pt_correlation_func_treecorr.pdf'))
