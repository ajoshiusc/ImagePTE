import glob
import os
import sys

import h5py
import numpy as np
#from dfsio import readdfs, writedfs
from scipy import io as spio
from tqdm import tqdm

#from surfproc import patch_color_attrib, smooth_surf_function
sys.path.append('../stats')

from brainsync import normalizeData
from get_connectivity import get_connectivity

#%%

if __name__ == "__main__":

    atlas_labels = '/home/ajoshi/projects/bfp/supp_data/USCBrain_grayordinate_labels.mat'

    atlas = spio.loadmat(atlas_labels)

    gord_labels = atlas['labels'].squeeze()

    label_ids = np.unique(gord_labels)  # unique label ids

    # remove WM label from connectivity analysis
    label_ids = np.setdiff1d(label_ids, (2000, 0))

    p_dir = '/data_disk/Beijing_Zang_bfp'
    lst = glob.glob(p_dir + '/*/func/*_rest_bold.32k.GOrd.mat')
    count1 = 0

    # Get number of subjects
    nsub = len(lst)

    conn_mat = np.zeros((len(label_ids), len(label_ids), nsub))

    for subno, subfile in enumerate(tqdm(lst)):
        head_tail = os.path.split(subfile)
        fname = os.path.join(subfile)

        f = spio.loadmat(fname)

        d = np.array(f['dtseries']).T

        # Get connectivity matrix for each subject
        conn_mat[:, :, subno] = get_connectivity(d,
                                                 labels=gord_labels,
                                                 label_ids=label_ids)

    input('press any key')
