import numpy as np
import scipy.stats as ss
from statsmodels.stats.multitest import fdrcorrection
import scipy as sp
import os
import sys
import nilearn.image as ni



BFPPATH='/ImagePTE1/ajoshi/code_farm/bfp'

sys.path.append(os.path.join(BFPPATH, 'src/stats/'))

from grayord_utils import vis_grayord_sigcorr, save2volbord_bci



nonepi_data = np.load('NONPTE_fmridiff_BOrd.npz')['fdiff_sub_z']
epi_data = np.load('PTE_fmridiff_BOrd.npz')['fdiff_sub_z']



F = epi_data.var(axis=1) / (nonepi_data.var(axis=1) + 1e-16)
save2volbord_bci(F, 'F.nii.gz', bfp_path=BFPPATH, smooth_std=1.0)

pval = 1-ss.f.cdf(F, 37 - 1, 37 - 1)

pval[np.isnan(pval)] = .5
_, pval_fdr = fdrcorrection(pval)

save2volbord_bci(pval, 'Fpval.nii.gz', bfp_path=BFPPATH, smooth_std=1.0)

save2volbord_bci(pval_fdr, 'Fpval_fdr.nii.gz', bfp_path=BFPPATH, smooth_std=1.0)
