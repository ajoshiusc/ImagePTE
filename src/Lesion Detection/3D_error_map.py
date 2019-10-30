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
import VAE_model_pixel64  as ProbVAE
import scipy.stats as st
from sklearn import metrics
import VAE_model_pixel_vanilla  as VAE
from data_save_VAE import reshape
pret=0

#p_values = scipy.stats.norm.sf(abs(z_scores))*2
    
def load_model(epoch, encoder, decoder, loc):
    #  restore models
    decoder.load_state_dict(torch.load(loc+'/VAE_GAN_decoder_%d.pth' % epoch))
    decoder.cuda()
    encoder.load_state_dict(torch.load(loc+'/VAE_GAN_encoder_%d.pth' % epoch))
    encoder.cuda()

def save_out(y_probas,num_C,ref_dir):
    t1file = t1_file = os.path.join(ref_dir, 'T1mni.nii.gz')
    t1model=ni.load_img(t1file )
    for i in range(y_probas,size[2]/num_C):
        subi=y_probas[:,:,i*num_C:(i+1)*num_C,:]
        img = ni.new_img_like(t1model, MSE_image)
        img.to_filename('/big_disk/akrami/MSE_nonep_%d.nii.gz' %i)

#####read data######################
d=np.load('data_maryland_128.npz')
X_valid=d['data']




d=np.load('data_TBI_128.npz')
X_valid=np.concatenate((X_train,d['data']),axis=0)

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
hidden_size = 128
max_epochs = 200
lr = 3e-4
beta = 0
device='cuda'
#########################################
epoch=39
LM='/models/VAE_final'

###### define constant########
input_channels = 3
hidden_size =128
max_epochs = 100
lr = 3e-4
beta =0
device='cuda'
#########################################
epoch=99
LM='models/VAE_final'

##########load low res net##########
G=VAE_Generator(input_channels, hidden_size).cuda()
load_model(epoch,G.encoder, G.decoder,LM)



##########define beta loss##########

def MSE_loss(Y, X):
    msk = torch.tensor(X > 1e-6).float()
    ret = ((X- Y) ** 2)*msk)
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
def Validation(X):
    G.eval()
#G2.eval()
    test_loss = 0
    ind = 0
    with torch.no_grad():
        for i, data in enumerate(Validation_loader_inference):
            data = (data).to(device)
            msk = torch.tensor(data > 1e-6).float()
            seg = X[ind:ind + batch_size, ::2, ::2, 3]
            ind = ind + batch_size
            seg = torch.from_numpy(seg)
            seg = (seg).to(device)
            _, _, arr_lowrec = G(data)
            f_recon_batch = arr_lowrec[:, 2, :, :]*msk[:, 2, :, :]

            

            f_data = data[:, 2, :, :]*msk[:, 2, :, :]
            #f_recon_batch = f_recon_batch[:, 2, :, :]
            rec_error = torch.abs(f_data - f_recon_batch)*msk[:, 2, :, :]
            if i==0:
                rec_error_all = rec_error
            else:
                rec_error_all = torch.cat([rec_error_all, rec_error])
    #test_loss /= len(Validation_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return rec_error_all


if __name__ == "__main__":
    sub_size=[182,218]
    num_C=182
    ref_dir='/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1/TBI_INVYU830PA1'
    y_probas = Validation(X)
    y_probas = (y_probas).to('cpu')
    y_probas = y_probas.numpy()
    y_probas=scipy.signal.medfilt(ry_probas,(1,1,7,7))
    y_probas = np.transpose(y_probas, (2, 3, 0,1))
    reshape(y_probas,sub_size,3)
    save_out(y_probas,num_C,ref_dir)

    
    

    



  