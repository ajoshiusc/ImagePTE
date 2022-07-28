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

config_file = '/home/ajoshi/projects/ImagePTE/src/stats/sample_config_stats_Maryland_ALFF.ini'
BFPPATH = '/home/ajoshi/projects/bfp'
# %%#%%
# Import the required librariesimport configparser

# get BFP directory from config file
config = configparser.ConfigParser()
config.read(config_file)
section = config.sections()
bfp_path = config.get('inputs', 'bfp_path')
sys.path.append(os.path.join(bfp_path, 'src/stats/'))
from stats_utils import randpair_groupdiff, randpair_groupdiff_ftest
from grayord_utils import vis_grayord_sigcorr, save2volbord_bci


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
subIDs, sub_fname, group, reg_var, reg_cvar1, reg_cvar2 = read_demoCSV(
    cf.csv_fname, cf.data_dir, cf.file_ext, cf.colsubj, cf.colvar_exclude,
    cf.colvar_group, cf.colvar_main, cf.colvar_reg1, cf.colvar_reg2, len_time=int(cf.lentime))

group = np.int16(group)
# for boolan indexing, need to convert to numpy array
subIDs = np.array(subIDs)
sub_fname = np.array(sub_fname)

print('Identifying subjects for each group...')
subIDs_grp2 = subIDs[group == 1]
sub_fname_grp2 = sub_fname[group == 1]

subIDs_grp1 = subIDs[group == 0]
sub_fname_grp1 = sub_fname[group == 0]

num_bord = 46961
# Read group 1
data1 = np.zeros((num_bord, len(subIDs_grp1)))
for i, fname in enumerate(sub_fname_grp1):
    data1[:, i] = spio.loadmat(fname)['dtseries'].squeeze()

data2 = np.zeros((num_bord, len(subIDs_grp2)))
for i, fname in enumerate(sub_fname_grp2):
    data2[:, i] = spio.loadmat(fname)['dtseries'].squeeze()


pval = np.zeros(num_bord)

# Ranksum test does not show any significance
# for i in range(num_bord):
#    _, pval[i]=ranksums(data1[i,:], data2[i,:])

# We will perform f-test test (modified in a pairwise stats)

S1 = 0.5 * np.var(data1, axis=1)
S2 = 0.5 * np.var(data2, axis=1)


n1 = data1.shape[1]
n2 = data2.shape[1]

F = S1 / (S2 + 1e-16)

pval = 1 - ss.f.cdf(F, n1 - 1, n2 - 1)

# %%
pval[sp.isnan(pval)] = .5
_, pval_fdr = fdrcorrection(pval)


save2volbord_bci(pval, os.path.join(bfp_path, 'src/stats/results',
                                    'pval_alff_bord_PTE_smooth0.5.nii.gz'), bfp_path=BFPPATH, smooth_std=0.5)
save2volbord_bci(pval_fdr, os.path.join(bfp_path, 'src/stats/results',
                                        'pval_alff_fdr_bord_PTE_smooth0.5.nii.gz'), bfp_path=BFPPATH, smooth_std=0.5)

save2volbord_bci((0.05-pval)*np.float32(pval < 0.05), os.path.join(bfp_path, 'src/stats/results',
                 'pval_alff_bord_PTE_smooth1.5_sig.nii.gz'), bfp_path=BFPPATH, smooth_std=1.5)
save2volbord_bci((0.05-pval_fdr)*np.float32(pval_fdr < 0.05), os.path.join(bfp_path,
                 'src/stats/results', 'pval_fdr_bord_PTE_smooth1.5_sig.nii.gz'), bfp_path=BFPPATH, smooth_std=1.5)
