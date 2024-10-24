## Lesion analysis for EPIBIOSX data

# Importing libraries
import os
import numpy as np
import pandas as pd
import nibabel as nib
import glob
from tqdm import tqdm

t2_path = '//deneb_disk/lesion_masks_reg/new_registered/t2flair/'

lesion_seg_path = '/deneb_disk/lesion_masks_reg/new_registered/total_lesion_masks/'

output_dir = '/deneb_disk/lesion_masks_reg/new_registered/output_dir'


pte_xlsx = os.path.join('spreadsheets/short PTE.xlsx')


# read the spreadsheet with the PTE data, and extract the subject IDs and PTE status
pte_data = pd.read_excel(pte_xlsx)
pte_data = pte_data[['Study ID', 'PTE', 'LS']]
pte_data = pte_data.dropna(subset=['Study ID'])




# Load the lesion segmentation data and calculate the average map of the lesion

lesion_seg_files = glob.glob(os.path.join(lesion_seg_path, '*.nii.gz'))
t1_flair_files = glob.glob(os.path.join(t2_path, '*.nii.gz'))

t2flair_data_LS = []
t2flair_data_noLS = []

for sub in pte_data['Study ID']:

    files = [f for f in t1_flair_files if sub in f]
    if len(files) == 0:
        continue

    if len(files) > 1:
        print('Multiple files found for subject {}'.format(sub))
        continue

    t2flair_data = []
    v = nib.load(files[0]).get_fdata()
    t2flair_data.append(v)
    myaffine = nib.load(files[0]).affine


    #print('Processing subject {}'.format(sub))
    print(pte_data[pte_data['Study ID'] == sub])
    if 'Y' in pte_data[pte_data['Study ID'] == sub]['LS']:
        t2flair_data_LS.append(v)
    else:
        t2flair_data_noLS.append(v)

t2flair_data_LS = np.array(t2flair_data_LS)
t2flair_data_noLS = np.array(t2flair_data_noLS)

t2flair_data_LS_mean = np.mean(t2flair_data_LS, axis=0)
t2flair_data_noLS_mean = np.mean(t2flair_data_noLS, axis=0)

t2flair_data_LS_mean = nib.Nifti1Image(t2flair_data_LS_mean, affine=myaffine)
nib.save(t2flair_data_LS_mean, os.path.join(output_dir, 't2flair_mean_LS.nii.gz'))

t2flair_data_noLS_mean = nib.Nifti1Image(t2flair_data_noLS_mean, affine=myaffine)
nib.save(t2flair_data_noLS_mean, os.path.join(output_dir, 't2flair_mean_noLS.nii.gz'))

# Do a ranksum test to see if the mean T2 FLAIR intensity is different between the two groups, and save results as an image of p values
from scipy.stats import ranksums

t2flair_data_LS_mean = t2flair_data_LS_mean.get_fdata()#.flatten()
t2flair_data_noLS_mean = t2flair_data_noLS_mean.get_fdata()#.flatten()

p_values = np.zeros(t2flair_data_LS_mean.shape)
p_values = p_values.reshape(t2flair_data_LS_mean.shape)

for i in range(t2flair_data_LS_mean.shape[0]):
    for j in range(t2flair_data_LS_mean.shape[1]):
        for k in range(t2flair_data_LS_mean.shape[2]):
            p_values[i, j, k] = ranksums(t2flair_data_LS_mean[i, j, k], t2flair_data_noLS_mean[i, j, k])[1]

p_values = nib.Nifti1Image(p_values, affine=myaffine)
nib.save(p_values, os.path.join(output_dir, 'p_values.nii.gz'))

