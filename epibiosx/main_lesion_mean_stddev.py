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

lesion_seg_data = []
for file in tqdm(lesion_seg_files):
    v = nib.load(os.path.join(lesion_seg_path, file)).get_fdata()
    lesion_seg_data.append(v)
    print(v.shape)

myaffine = nib.load(os.path.join(lesion_seg_path, file)).affine
lesion_seg_data = np.array(lesion_seg_data)
lesion_seg_data = np.mean(lesion_seg_data, axis=0)

# Save the average lesion map
lesion_seg_avg = nib.Nifti1Image(lesion_seg_data, affine=myaffine)
nib.save(lesion_seg_avg, os.path.join(output_dir, 'average_lesion_seg.nii.gz'))

# Load the T2 FLAIR data and calculate the mean and standard deviation of the lesion
t2_files = glob.glob(os.path.join(t2_path, '*.nii.gz'))

t2_data = []
for file in tqdm(t2_files):
    v = nib.load(os.path.join(t2_path, file)).get_fdata()
    t2_data.append(v)
    print(v.shape)

t2_data = np.array(t2_data)
t2_data_mean = np.mean(t2_data, axis=0)
t2_data_std = np.std(t2_data, axis=0)

# Save the mean and standard deviation of the T2 FLAIR data
t2_data_mean = nib.Nifti1Image(t2_data_mean, affine=myaffine)
nib.save(t2_data_mean, os.path.join(output_dir, 't2flair_mean.nii.gz'))



# plot overlay of the average lesion map on the mean T2 FLAIR image using nilearn library's plot_stat_map function
from nilearn import plotting

plotting.plot_stat_map(lesion_seg_avg, bg_img=t2_data_mean)

# cut_coords=None, output_file=os.path.join(output_dir, 't2flair_mean_lesion_overlay.png'), display_mode='ortho', colorbar=True, title='T2 FLAIR mean with lesion overlay', draw_cross=False, annotate=True, black_bg=False, threshold=None, bg_vmin=None, bg_vmax=None, cmap='viridis', dim='auto', vmax=None, resampling_interpolation='continuous', symmetric_cbar='auto', cbar_vmin=None, cbar_vmax=None, output_format='png', figure=None, axes=None, title_font_size=12, annotate_font_size=10, draw_midline=False, cut_coords_text=None, black_bg_style='light', threshold_mask=None, mask_args=None, alpha=0.5)

#plotting.plot_stat_map(lesion_seg_avg, bg_img=t2_data_mean, cut_coords=None, output_file=os.path.join(output_dir, 't2flair_mean_lesion_overlay.png'), display_mode='ortho', colorbar=True, title='T2 FLAIR mean with lesion overlay', draw_cross=False, annotate=True, black_bg=False, threshold=None, bg_vmin=None, bg_vmax=None, cmap='viridis', dim='auto', vmax=None, resampling_interpolation='continuous', symmetric_cbar='auto', cbar_vmin=None, cbar_vmax=None, output_format='png', figure=None, axes=None, title_font_size=12, annotate_font_size=10, draw_midline=False, cut_coords_text=None, black_bg_style='light', threshold_mask=None, mask_args=None, alpha=0.5)

plotting.show()





# plot the standard deviation of the T2 FLAIR data


