# %%
#config_file = '/home/ajoshi/coding_ground/bfp/src/stats/sample_config_stats_ADHD.ini'
import scipy.stats as ss
from read_data_utils import load_bfp_data, read_demoCSV, write_text_timestamp, readConfig, read_demoCSV_list
from statsmodels.stats.multitest import fdrcorrection
import configparser
import numpy as np
import scipy as sp
import scipy.io as spio
import os
import sys
from scipy.stats import ranksums
from numpy.random import shuffle
from tqdm import tqdm

config_file = '/home/ajoshi/Projects/ImagePTE/src/stats/sample_config_stats_Maryland_ALFF_hbm2024.ini'
BFPPATH = '/home/ajoshi/Projects/bfp'
# %%#%%
# Import the required librariesimport configparser

# get BFP directory from config file
config = configparser.ConfigParser()
config.read(config_file)
section = config.sections()
bfp_path = config.get('inputs', 'bfp_path')
sys.path.append(os.path.join(bfp_path, 'src/stats/'))

from grayord_utils import vis_grayord_sigcorr, save2volbord_bci
from stats_utils import randpair_groupdiff, randpair_groupdiff_ftest



def permutation_Ftest(x, y, nperm=1000):
    # x,y are assumed to be subjects x features arrays

    F_orig = x.var(axis=0) / (y.var(axis=0) + 1e-16)
    #F_orig = kurtosis(x)/kurtosis(y)

    # concatenate x and y data to get subjects x features
    data = np.concatenate((x, y), axis=0)

    print('Permutation testing')
    max_null = np.zeros(nperm)
    n_count = 0

    for ind in tqdm(range(nperm)):
        shuffle(data)  # shuffle along subjects axis
        x_perm = data[:x.shape[0], ]
        y_perm = data[x.shape[0]:, ]

        F_perm = x_perm.var(axis=0) / (y_perm.var(axis=0) + 1e-16)
        #F_perm = kurtosis(x_perm) / kurtosis(y_perm)

        max_null[ind] = np.amax(F_perm)
        n_count += np.float32(F_perm > F_orig)

    pval_max = np.sum(F_orig[:, None] < max_null[None, :], axis=1) / nperm

    pval_perm = n_count / (nperm)

    _, pval_perm_fdr = fdrcorrection(pval_perm)

    return pval_max, pval_perm_fdr, pval_perm, F_orig





os.chdir(bfp_path)
cf = readConfig(config_file)

# Import BrainSync libraries

# %%
log_fname = os.path.join(cf.out_dir, 'bfp_group_stat_log.txt')
write_text_timestamp(log_fname, 'Config file used: ' + config_file)
if not os.path.isdir(cf.out_dir):
    os.makedirs(cf.out_dir)
write_text_timestamp(log_fname,
                     "All outputs will be written in: " + cf.out_dir)
# read demographic csv file
subIDs, sub_fname, ref_atlas, group, reg_cvar1, reg_cvar2 = read_demoCSV(
    cf.csv_fname, cf.data_dir, cf.file_ext, cf.colsubj, cf.colvar_exclude,
    cf.colvar_group, cf.colvar_main, cf.colvar_reg1, cf.colvar_reg2, len_time=int(cf.lentime))

group = np.int16(group)
# for boolan indexing, need to convert to numpy array
subIDs = np.array(subIDs)
sub_fname = np.array(sub_fname)

print('Identifying subjects for each group...')

subIDs_grp_pte = subIDs[group == 1]
sub_fname_grp2 = sub_fname[group == 1]

subIDs_grp_nonpte = subIDs[group == 0]
sub_fname_grp1 = sub_fname[group == 0]

num_bord = 46961
# Read group 1
data1 = np.zeros((num_bord, len(subIDs_grp_nonpte)))
for i, fname in enumerate(sub_fname_grp1):
    data1[:, i] = spio.loadmat(fname)['dtseries'].squeeze()

data2 = np.zeros((num_bord, len(subIDs_grp_pte)))
for i, fname in enumerate(sub_fname_grp2):
    data2[:, i] = spio.loadmat(fname)['dtseries'].squeeze()


pval = np.zeros(num_bord)

# Ranksum test does not show any significance
# for i in range(num_bord):
#    _, pval[i]=ranksums(data1[i,:], data2[i,:])

# We will perform f-test test (modified in a pairwise stats)
#data1 = data1[:,:35]

S1 = 0.5 * np.var(data1, axis=1)
S2 = 0.5 * np.var(data2, axis=1)


n1 = data1.shape[1]
n2 = data2.shape[1]

F = S2 / (S1 + 1e-16)

pval = 1 - ss.f.cdf(F, n2 - 1, n1 - 1)

# %%
pval[np.isnan(pval)] = .5
_, pval_fdr = fdrcorrection(pval)


pval_max, pval_fdr, pval, Fval = permutation_Ftest(data2.T, data1.T, nperm=1000)

outpath = '/home/ajoshi/Projects/ImagePTE/src/alff_analysis'

save2volbord_bci((0.05-pval)*np.float32(pval < 0.05), os.path.join(outpath,'pval_alff_bord_PTE_smooth0_sig_perm.nii.gz'), bfp_path=BFPPATH, smooth_std=0)
save2volbord_bci((0.05-pval_fdr)*np.float32(pval_fdr < 0.05), os.path.join(outpath, 'pval_fdr_bord_PTE_smooth0_sig_perm.nii.gz'), bfp_path=BFPPATH, smooth_std=0)

save2volbord_bci((0.05-pval)*np.float32(pval < 0.05), os.path.join(outpath,'pval_alff_bord_PTE_smooth0.5_sig_perm.nii.gz'), bfp_path=BFPPATH, smooth_std=.5)
save2volbord_bci((0.05-pval_fdr)*np.float32(pval_fdr < 0.05), os.path.join(outpath, 'pval_fdr_bord_PTE_smooth0.5_sig_perm.nii.gz'), bfp_path=BFPPATH, smooth_std=.5)

save2volbord_bci((0.05-pval)*np.float32(pval < 0.05), os.path.join(outpath,'pval_alff_bord_PTE_smooth1_sig_perm.nii.gz'), bfp_path=BFPPATH, smooth_std=1)
save2volbord_bci((0.05-pval_fdr)*np.float32(pval_fdr < 0.05), os.path.join(outpath, 'pval_fdr_bord_PTE_smooth1_sig_perm.nii.gz'), bfp_path=BFPPATH, smooth_std=1)

save2volbord_bci(pval, os.path.join(outpath,'pval2_alff_bord_PTE_smooth0_sig_perm.nii.gz'), bfp_path=BFPPATH, smooth_std=0)
save2volbord_bci(pval_fdr, os.path.join(outpath,'pval2_fdr_bord_PTE_smooth0_sig_perm.nii.gz'), bfp_path=BFPPATH, smooth_std=0)
save2volbord_bci(Fval, os.path.join(outpath, 'fval2_bord_PTE_smooth0_sig_perm.nii.gz'), bfp_path=BFPPATH, smooth_std=0)


save2volbord_bci(pval, os.path.join(outpath,'pval2_alff_bord_PTE_smooth0.5_sig_perm.nii.gz'), bfp_path=BFPPATH, smooth_std=.5)
save2volbord_bci(pval_fdr, os.path.join(outpath,'pval2_fdr_bord_PTE_smooth0.5_sig_perm.nii.gz'), bfp_path=BFPPATH, smooth_std=.5)
save2volbord_bci(Fval, os.path.join(outpath, 'fval2_bord_PTE_smooth0.5_sig_perm.nii.gz'), bfp_path=BFPPATH, smooth_std=.5)


save2volbord_bci(pval, os.path.join(outpath,'pval2_alff_bord_PTE_smooth1.5_sig_perm.nii.gz'), bfp_path=BFPPATH, smooth_std=1.5)
save2volbord_bci(pval_fdr, os.path.join(outpath,'pval2_fdr_bord_PTE_smooth1.5_sig_perm.nii.gz'), bfp_path=BFPPATH, smooth_std=1.5)
save2volbord_bci(Fval, os.path.join(outpath,'fval2_bord_PTE_smooth1.5_sig_perm.nii.gz'), bfp_path=BFPPATH, smooth_std=1.5)
