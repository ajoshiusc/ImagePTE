from __future__ import print_function
import numpy as np
import pywt
from matplotlib import pyplot as plt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis
import argparse
import h5py
import numpy as np
import os
import time
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image
import torchvision.utils as vutils
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats as st
from sklearn import metrics
from VAE_model_pixel_vanilla import Encoder, Decoder, VAE_Generator
import nilearn.image as ni
import cv2
import matplotlib.pyplot as plt

pret=0

#p_values = scipy.stats.norm.sf(abs(z_scores))*2
    
def load_model(epoch, encoder, decoder, loc):
    #  restore models
    decoder.load_state_dict(torch.load(loc+'/VAE_GAN_decoder_%d.pth' % epoch))
    decoder.cuda()
    encoder.load_state_dict(torch.load(loc+'/VAE_GAN_encoder_%d.pth' % epoch))
    encoder.cuda()

def save_out(MSE_image,num_C,ref_dir):
    t1file = t1_file = os.path.join(ref_dir, 'T1mni.nii.gz')
    t1model=ni.load_img(t1file )
    for i in range (int(MSE_image.shape[2]/num_C)):
        subi=MSE_image[:,:,i*num_C:(i+1)*num_C,2]
        img = ni.new_img_like(t1model, subi)
        img.to_filename('/big_disk/akrami/git_repos_new/ImagePTE/src/Lesion Detection/models/3D_out_R/MSE_FLAIR_%d.nii.gz' %i)
        subi=MSE_image[:,:,i*num_C:(i+1)*num_C,0]
        img = ni.new_img_like(t1model, subi)
        img.to_filename('/big_disk/akrami/git_repos_new/ImagePTE/src/Lesion Detection/models/3D_out_R/MSE_T1_%d.nii.gz' %i)
        subi=MSE_image[:,:,i*num_C:(i+1)*num_C,1]
        img = ni.new_img_like(t1model, subi)
        img.to_filename('/big_disk/akrami/git_repos_new/ImagePTE/src/Lesion Detection/models/3D_out_R/MSE_T2_%d.nii.gz' %i)

def reshape(data_In,size,num_channel):
   data=data_In
   X=np.zeros((data.shape[0],size[0], size[1],num_channel))
   for i in range(data.shape[0]):
      if i==0:
         X[i,:,:,:]= cv2.resize(data[i,:,:,:], dsize=(size[1], size[0]), interpolation=cv2.INTER_CUBIC)
         #X=X.reshape((1,size[0],size[1],num_channel))
      else:
         X[i,:,:,:]=cv2.resize(data[i,:,:,:], dsize=(size[1], size[0]), interpolation=cv2.INTER_CUBIC) 
         #temp=temp.reshape((1,size[0],size[1],num_channel))
         #X=np.append(X,temp ,axis=0)
         
   fig, ax = plt.subplots()
   im = ax.imshow(X[50,:,:,0])
   plt.show()
   return X
#####read data######################
d=np.load('./data/data_maryland_128_pilepsy_test.npz')
X_valid=d['data']




d=np.load('./data/data_maryland_128_nonepilepsy_test.npz')
X_valid=np.concatenate((X_valid,d['data']),axis=0)
X_valid=X_valid[:,:,:,:]

####################################



##########train validation split##########
batch_size=8


X_valid = np.transpose(X_valid, (0, 3, 1,2))
validation_data_inference = torch.from_numpy(X_valid).float()
validation_data_inference= validation_data_inference.to('cuda') 


Validation_loader = torch.utils.data.DataLoader(validation_data_inference,
                                          batch_size=batch_size,
                                          shuffle=False)
                                         
############################################



########## intilaize parameters##########        
# define constant
input_channels = 3
hidden_size = 64
max_epochs = 200
lr = 3e-4
beta = 0
device='cuda'
#########################################
epoch=99
LM='/big_disk/akrami/git_repos_new/ImagePTE/src/Lesion Detection/models/RVAE_final_1'

##########load low res net##########
G=VAE_Generator(input_channels, hidden_size).cuda()
load_model(epoch,G.encoder, G.decoder,LM)




##########define beta loss##########

def MSE_loss(Y, X):
    msk = torch.tensor(X > 1e-6).float()
    ret = ((X- Y) ** 2)*msk
    return ret 
def BMSE_loss(Y, X, beta,sigma,Dim):
    term1 = -((1+beta) / beta)
    K1=1/pow((2*math.pi*( sigma** 2)),(beta*Dim/2))
    term2=MSE_loss(Y, X)
    term3=torch.exp(-(beta/(2*( sigma** 2)))*term2)
    loss1=torch.sum(term1*(K1*term3-1))
    return loss1



# Reconstruction + KL divergence losses summed over all elements and batch

def beta_loss_function(recon_x, x, mu, logvar, beta):

    if beta > 0:
        sigma=1
        # If beta is nonzero, use the beta entropy
        BBCE = BMSE_loss(recon_x.view(-1, 128*128*1), x.view(-1, 128*128*1), beta,sigma,128*128*1)
    else:
        # if beta is zero use binary cross entropy
        BBCE = torch.sum(MSE_loss(recon_x.view(-1, 128*128*1),x.view(-1, 128*128*1)))

    # compute KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BBCE +KLD

####################################

##########TEST##########
def Validation():
    G.eval()
#G2.eval()
    test_loss = 0
    ind = 0
    with torch.no_grad():
        for i, data in enumerate(Validation_loader):
            data = (data).to(device)
            msk = torch.tensor(data > 1e-6).float()
            _, _, arr_lowrec = G(data)
            f_recon_batch = arr_lowrec[:, :, :, :]*msk[:, :, :, :]

            

            f_data = data[:, :, :, :]*msk[:, :, :, :]
            #f_recon_batch = f_recon_batch[:, 2, :, :]
            rec_error = torch.abs(f_data - f_recon_batch)*msk[:, :, :, :]
            if i==0:
                rec_error_all = rec_error
            else:
                rec_error_all = torch.cat([rec_error_all, rec_error])
    #test_loss /= len(Validation_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return rec_error_all

sub_size=[182,218]
num_C=182
ref_dir='/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1/TBI_INVYU830PA1'
y_probas = Validation()
y_probas = (y_probas).to('cpu')
y_probas = y_probas.numpy()
y_probas=scipy.signal.medfilt(y_probas,(1,1,7,7))
y_probas = np.transpose(y_probas, (0,2, 3, 1))
y_probas=reshape(y_probas,sub_size,3)
y_probas = np.transpose(y_probas, (1,2, 0, 3))
save_out(y_probas,num_C,ref_dir)

    
    

    



  