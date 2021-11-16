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
from tqdm import tqdm
from sklearn.svm import OneClassSVM
pret=0

#p_values = scipy.stats.norm.sf(abs(z_scores))*2
    
def load_model(epoch, encoder, decoder, loc):
    #  restore models
    decoder.load_state_dict(torch.load(loc+'/VAE_GAN_decoder_%d.pth' % epoch))
    decoder.cuda()
    encoder.load_state_dict(torch.load(loc+'/VAE_GAN_encoder_%d.pth' % epoch))
    encoder.cuda()
  
def dice_coef(mask1, mask2):
    mask1 = np.float32(mask1)
    mask2 = np.float32(mask2)

    return 2 * np.sum(
        mask1 * mask2) / (np.sum(np.float32(mask1 + mask2)) + 1e-8)

#####read data######################

d=np.load('/ImagePTE1/akrami/Projects/crash_gits/ImagePTE/src/Lesion Detection/data/data_ISEL_128_valid.npz')
X = d['data'][:,:,:,:]

X_data = np.copy(X[:, :, :, 0:3])
#max_val=np.max(X)
#X_data = X_data/ max_val
X_data = X_data.astype('float64')
X_valid=X_data[:,:,:,:]
D=X_data.shape[1]*X_data.shape[2]
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
LM='/ImagePTE1/akrami/Projects/crash_gits/ImagePTE/src/Lesion Detection/models/VAE_final_64'

##########load low res net##########
G=VAE_Generator(input_channels, hidden_size).cuda()
load_model(epoch,G.encoder, G.decoder,LM)







##########define beta loss##########

def MSE_loss(Y, X):
    msk = torch.tensor(X > 1e-6).float()
    ret = ((X- Y) ** 2)*msk
    ret = torch.sum(ret,1)
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
def Validation_2(X):
    G.eval()
#G2.eval()
    test_loss = 0
    ind = 0
    with torch.no_grad():
        for i, data in enumerate(Validation_loader):
            data = (data).to(device)
            msk = torch.tensor(data > 1e-6).float()
            seg = X[ind:ind + batch_size, :, :, 3]
            seg=seg.astype('float32')
            ind = ind + batch_size
            seg = torch.from_numpy(seg)
            seg = (seg).to(device)*msk[:, 2, :, :]
            _, _, arr_lowrec = G(data)
            f_recon_batch = arr_lowrec[:, 2, :, :]*msk[:, 2, :, :]

            

            f_data = data[:, 2, :, :]*msk[:, 2, :, :]
            #f_recon_batch = f_recon_batch[:, 2, :, :]
            rec_error =(f_data - f_recon_batch)*msk[:, 2, :, :]
            #rec_error=torch.mean(rec_error,1)
            if i<200:
                n = min(f_data.size(0), 100)
                err=torch.abs(f_data.view(batch_size,1, 128, 128)[:n] -
                     f_recon_batch.view(batch_size,1, 128, 128)[:n])
                err=err
                #err=torch.mean(err,1)
                median=(err).to('cpu')
                median=median.numpy()
                median=scipy.signal.medfilt(median,(1,1,7,7))
                median=median.astype('float32')
                median = np.clip(median, 0, 1)
    
                err=median
                err=torch.from_numpy(err)
                err=(err).to(device)

                comparison = torch.cat([
                    f_data.view(batch_size, 1, 128,128)[:n],
                    f_recon_batch.view(batch_size, 1, 128, 128)[:n],
                    err.view(batch_size, 1, 128, 128)[:n],
                    torch.abs(
                        f_data.view(batch_size, 1, 128, 128)[:n] -
                        f_recon_batch.view(batch_size, 1, 128, 128)[:n]),
                    seg.view(batch_size, 1, 128, 128)[:n]
                ])
                save_image(comparison.cpu(),
                           './results/reconstruction_isel' +str(i)+ '.png',
                           nrow=n)
                
            if i==0:
                rec_error_all = rec_error
            else:
                rec_error_all = torch.cat([rec_error_all, rec_error])
    #test_loss /= len(Validation_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return rec_error_all



rec_error_all = Validation_2(X)
y_true =(X[:, :, :, 3])


maskX = np.copy(X[:, :, :, 2])
y_true[y_true > 0]=1
y_true[y_true < 0]=0
#y_true = y_true[maskX > 0]

y_probas = (rec_error_all).to('cpu')
y_probas = y_probas.numpy()

y_true = y_true.astype(int)
#y_probas = np.clip(y_probas, 0, 1)





###
indx=[]
for i in range(y_true.shape[0]):
    if y_true[i,:,:].mean()!=0:
        indx.append(i)


y_probas= np.reshape(y_probas, (-1,128*128))
y_true=y_true[indx,:,:]
y_probas=y_probas[indx,:]
Xout = np.zeros(y_probas.shape)
y_probas=y_probas.T
maskX= maskX[indx,:,:]


#temp= OneClassSVM(gamma='scale').fit_predict(y_probas[[0],:].T)
for j in tqdm(range(y_probas.shape[1])):
    temp1=y_probas[:,j:j+1]
    if temp1.mean()==0:
        continue
    Xout[j, ] = OneClassSVM(gamma=0.001).fit_predict(temp1)== -1
###
#y_probas=scipy.signal.medfilt(y_probas,(1,1,7,7))
Xout = np.reshape(Xout, (-1, 128,128))
Xout[maskX <= 0]=0

out_dice=np.zeros((y_true.shape[0],1))

for i in range(y_true.shape[0]):
    out_dice[i] = dice_coef(Xout[i,:,:], y_true[i,:,:])

print(np.mean(out_dice, axis=0))
print(np.std(out_dice, axis=0))


    
   
  
    
 
    
