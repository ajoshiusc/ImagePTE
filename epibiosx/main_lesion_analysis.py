## Lesion analysis for EPIBIOSX data

# Importing libraries
import os
import numpy as np
import pandas as pd
import nibabel as nib
import glob
from tqdm import tqdm

epibiosx_path = '/deneb_disk/ifs/loni/faculty/dduncan/rgarner/shared/epibios'

lesion_seg_path = os.path.join(epibiosx_path, 'human/data/lesion_segmentations_box/total_lesion_masks/second_review')



pte_xlsx = os.path.join('spreadsheets/short PTE.xlsx')


# read the spreadsheet with the PTE data, and extract the subject IDs and PTE status
pte_data = pd.read_excel(pte_xlsx)
pte_data = pte_data[['Study ID', 'PTE']]
pte_data = pte_data.dropna(subset=['Study ID'])




# Load the lesion segmentation data and calculate the average map of the lesion

lesion_seg_files = glob.glob(os.path.join(lesion_seg_path, '*.nii.gz'))

lesion_seg_data = []
for file in tqdm(lesion_seg_files):
    v = nib.load(os.path.join(lesion_seg_path, file)).get_fdata()
    lesion_seg_data.append(v)
    print(v.shape)

lesion_seg_data = np.array(lesion_seg_data)
lesion_seg_data = np.mean(lesion_seg_data, axis=0)

# Save the average lesion map
lesion_seg_avg = nib.Nifti1Image(lesion_seg_data, affine=None)
nib.save(lesion_seg_avg, os.path.join(lesion_seg_path, 'average_lesion_seg.nii.gz'))

# Display the average lesion map
import nilearn.plotting as nip

nip.plot_roi(lesion_seg_avg, title='Average Lesion Segmentation', display_mode='z', cut_coords=5, draw_cross=False, cmap='hot', alpha=0.5)

# save to png file
nip.plot_roi(lesion_seg_avg, title='Average Lesion Segmentation', display_mode='z', cut_coords=5, draw_cross=False, cmap='hot', alpha=0.5, output_file='average_lesion_seg.png')





