import nilearn.image
import sys
import numpy as np
import matplotlib.pyplot as plt
from datautils_VAE import read_data, read_data_test
import cv2

###This fuction is for reading MRI data### 
def train_save(data_dir,ref_dir,sub_names,Out_name,window_H,window_W,window_C):
    data_dir =data_dir
    with open(sub_names) as f:
        tbidoneIds = f.readlines()
    tbidoneIds = [l.strip('\n\r') for l in tbidoneIds]

   
    slicerange = np.arange(0, window_C, dtype=int)

    data = read_data(study_dir=data_dir,
                                       ref_dir=ref_dir,
                                       subids=tbidoneIds,
                                       nsub=253,
                                       psize=[window_H, window_W],
                                       npatch_perslice=1,
                                       slicerange=slicerange,
                                       erode_sz=0,
                                       lesioned=False,
                                       dohisteq=True
                                       )
    fig, ax = plt.subplots()
    im = ax.imshow(data[10, :, :, 2])
    plt.show()
    np.savez(Out_name, data=data)
############################################

###This fuction is for reshaping MRI data### 
def reshape(data_In,size,num_channel):
   d=np.load(data_In)
   data=d['data']
   for i in range(data.shape[0]):
      if i==0:
         X= cv2.resize(data[i,:,:,:], dsize=(size[0], size[1]), interpolation=cv2.INTER_CUBIC)
         X=X.reshape((1,size[0],size[1],num_channel))
      else:
         temp=cv2.resize(data[i,:,:,:], dsize=(size[0], size[1]), interpolation=cv2.INTER_CUBIC) 
         temp=temp.reshape((1,size[0],size[1],num_channel))
         X=np.append(X,temp ,axis=0)
   return X
############################################

if __name__ == "__main__":
    ###data and output directory for merryland dataset###
    data_dir='/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1/'
    sub_names='/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs_training.txt'
    Out_name='data_maryland_128.npz'
    ref_dir='/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1/TBI_INVYU830PA1'

    ###initial parameters###
    window_H = 182
    window_W = 218
    window_C= 182

    ###read data###
    train_save(data_dir,ref_dir,sub_names,Out_name,window_H,window_W,window_C)

    ###reshape data###
    reshape_size=[128,128]
    num_channels=3     ##FLAIR T1 T2
    X_reshape=reshape(Out_name,reshape_size,num_channels)
    fig, ax = plt.subplots()
    im = ax.imshow(X_reshape[0,:,:,0])
    plt.show()
    np.savez(Out_name, data=X_reshape)       