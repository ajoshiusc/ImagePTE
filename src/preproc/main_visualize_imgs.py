import os
import nilearn.image as ni
from nilearn import plotting
import numpy as np


preproc_dir = '/big_disk/ajoshi/ISLES2015/preproc/Training'
subid = 1
dirname = os.path.join(preproc_dir, str(subid))

# Three modalities of MRI data for the given subject
fnamet1 = os.path.join(dirname, 'T1mni.nii.gz')
fnamet2 = os.path.join(dirname, 'T2mni.nii.gz')
fnameflair = os.path.join(dirname, 'FLAIRmni.nii.gz')

# Segmentation
fnameseg = os.path.join(dirname, 'SEGMENTATIONmni.nii.gz')

# View the overlay of T1 image and thresholded FLAIR image
view = plotting.view_img(stat_map_img=fnameflair,
                         bg_img=fnamet1, threshold=850)
view.open_in_browser()


# To get matrices of the images
t1 = ni.load_img(fnamet1)

t1_data = np.array(t1.get_data())

# Check the shape of the t1 matrix
print('Shape of the t1 data matrices is: ', t1_data.shape)

# To get matrices of the t2 images
t2 = ni.load_img(fnamet2)
t2_data = np.array(t2.get_data())
print('Shape of the t2 data matrices is: ', t2_data.shape)

# To get matrices of the flair images
flair = ni.load_img(fnameflair)
flair_data = np.array(flair.get_data())

# To get matrices of the segmentation images
seg = ni.load_img(fnameseg)
seg_data = np.array(seg.get_data())
