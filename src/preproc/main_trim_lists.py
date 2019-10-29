# coding: utf-8

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajoshiusc/lesion-detector/blob/master/main_anatomy_map.ipynb)

# In[1]:
import numpy as np
import nilearn.image as ni
import os

study_dir = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1'
pte_list = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy.txt'
tbi_list = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1_done.txt'
out_list = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs.txt'

#study_dir = '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot'
#pte_list = '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot_epilepsy.txt'
#tbi_list = '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot_done.txt'
#out_list = '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot_nonepilepsy_imgs.txt'

outfile = open(out_list, 'w')

with open(tbi_list) as f:
    tbiIds = f.readlines()

with open(pte_list) as f:
    pteIds = f.readlines()

# Get the list of pte subjects
pteIds = [l.strip('\n\r') for l in pteIds]
pteIds = list(set(pteIds))

# Get the list of tbi subjects
tbiIds = [l.strip('\n\r') for l in tbiIds]
tbiIds = list(set(tbiIds))

nonpteIds = list(set(tbiIds) - set(pteIds))

subDone = list()

for subj in nonpteIds:

    t1_file = os.path.join(study_dir, subj, 'T1mni.nii.gz')
    t1_mask_file = os.path.join(study_dir, subj, 'T1mni.mask.nii.gz')
    t2_file = os.path.join(study_dir, subj, 'T2mni.nii.gz')
    flair_file = os.path.join(study_dir, subj, 'FLAIRmni.nii.gz')

    if not (os.path.isfile(t1_file) and os.path.isfile(t1_mask_file)
            and os.path.isfile(t2_file) and os.path.isfile(flair_file)):
        continue

    outfile.write('%s\n' % subj)
    print(subj)

outfile.close()
