#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import net as nets
import train as tr
import utils as ut
import scipy.signal
from scipy import linalg
from scipy import spatial
import pathlib
import shutil
import os
import pandas as pd
import time
import warnings
import sys
from PIL import Image
import pickle
import imageio.v2 as iio
from tqdm import tqdm

from DISTS_pytorch import DISTS
import lpips

#%%

class JSD_NSS():
    def __init__(self):
        self.w = np.zeros((7,7))
        for i in range(7):
            for j in range(7):
                self.w[i,j] = np.exp(-((i-3)**2+(j-3)**2)/2)
        self.w = self.w / np.sum(self.w)

    def nss(self, input_img):
        mu = scipy.signal.convolve2d(input_img, self.w, mode='same')
        sigma = np.sqrt(scipy.signal.convolve2d((input_img-mu)**2, self.w, mode='same'))
        i_hat = (input_img - mu)/(sigma + 1)
        i_pmscn = i_hat[:-1,:-1]*i_hat[1:,1:]
        return i_pmscn

    def __call__(self, img0, img1):
        i0 = self.nss(img0).flatten()
        i1 = self.nss(img1).flatten()
        p,_ = np.histogram(i0,bins=np.linspace(-3,3,150))
        q,_ = np.histogram(i1,bins=np.linspace(-3,3,150)) 
        jsd = spatial.distance.jensenshannon(p,q)**2
        return jsd

class SSIM():
    def __init__(self, size = 11):
        self.window = np.ones((size, size))
        for i in range(size):
            for j in range(size):
                self.window[i,j] = np.exp(-(((i-size//2)**2+(j-size//2)**2)/(2*(1.5**2))))
        self.window /= self.window.sum()
                
    def __call__(self, img1, img2):
        
        kernel = self.window
        
        mean1 = scipy.signal.convolve2d(img1, kernel, mode='same', boundary = 'symm')
        mean2 = scipy.signal.convolve2d(img2, kernel, mode='same', boundary = 'symm')
        
        std1 = np.sqrt(scipy.signal.convolve2d((img1-mean1)**2, kernel, mode='same', boundary = 'symm'))
        std2 = np.sqrt(scipy.signal.convolve2d((img2-mean2)**2, kernel, mode='same', boundary = 'symm'))
        
        covar = scipy.signal.convolve2d((img2-mean2)*(img1-mean1), kernel, mode='same', boundary = 'symm')
        
        c1 = 0.01**2
        c2 = 0.03**2
        
        ssim = ((2*mean1*mean2 + c1)*(2*covar + c2))/((mean1**2 + mean2**2 + c1)*(std1**2 + std2**2 + c2))
        
        return np.mean(ssim)

class LPIPS():
    def __init__(self):
        self.loss = lpips.LPIPS(net='alex')
        
    def __call__(self, img1, img2):
        return self.loss(img1,img2).detach().item()

class InceptionV3(nn.Module):

    def __init__(self,
                 normalize_input=True,
                 requires_grad=False):

        super(InceptionV3, self).__init__()

        self.normalize_input = normalize_input
        self.blocks = nn.ModuleList()

        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            ]
        # Block 1: maxpool1 to maxpool2
        block1 = [
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
        ]
        
        self.blocks.append(nn.Sequential(*block0))
        self.blocks.append(nn.Sequential(*block1))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        outp = []
        x = inp

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx == 1:
                outp.append(x)

        return outp

class SIFID():
    def __init__(self, suffix = 'png'):
        self.suffix = suffix

    def get_activations(self, files, model, dims=192):
        model.eval()
        
        pred_arr = np.empty((len(files), dims))
    
        for i in range(len(files)):
    
            image = np.array([iio.imread(str(files[i])).astype(np.float32)])
            if len(image.shape) == 3:
                image = np.expand_dims(image,-1).repeat(3,-1)/255.
            elif len(image.shape) == 4:
                image = image[:,:,:,0:3]/255.
    
            # Reshape to (n_images, 3, height, width)
            image = image.transpose((0, 3, 1, 2))
            
            if np.max(image)>2:
                image /= 255
    
            batch = torch.from_numpy(image).type(torch.FloatTensor)
            pred = model(batch)[0]
    
            pred_arr = pred.cpu().data.numpy().transpose(0, 2, 3, 1).reshape(pred.shape[2]*pred.shape[3],-1)
    
        return pred_arr
    
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
    
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
    
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
    
        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'
    
        diff = mu1 - mu2
    
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real
    
        tr_covmean = np.trace(covmean)
    
        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)
    
    
    def calculate_activation_statistics(self, files, model, dims=192):
        
        act = self.get_activations(files, model, dims)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma
    
    
    def calculate_sifid_given_paths(self, path1, path2, dims, suffix):
        """Calculates the SIFID of two paths"""
    
        model = InceptionV3()
    
        path1 = pathlib.Path(path1)
        files1 = sorted(list(path1.glob('*.%s' %suffix)))
    
        path2 = pathlib.Path(path2)
        files2 = sorted(list(path2.glob('*.%s' %suffix)))
    
        fid_values = []
    
        for i in range(len(files2)):
            m1, s1 = self.calculate_activation_statistics([files1[i]], model, dims)
            m2, s2 = self.calculate_activation_statistics([files2[i]], model, dims)
            fid_values.append(self.calculate_frechet_distance(m1, s1, m2, s2))
           
        return fid_values

    def __call__(self, path1, path2):

        sifid_values = self.calculate_sifid_given_paths(path1, path2, 192, self.suffix)
        sifid_values = np.asarray(sifid_values,dtype=np.float32)
        
        return sifid_values.mean()

def torch2img(tensor):
    #Takes as entry a 4d tensor of shape (bsize,channels,width,height) 
    #and returns a 3d array of shape (width,height,channels)
    if len(tensor.shape) == 4:
        return tensor[0].permute(1,2,0).numpy()
    elif len(tensor.shape) == 3:
        return tensor.permute(1,2,0).numpy()
    else:
        raise ValueError('tensor has wrong shape!')
    
def img2torch(img):
    #Takes as entry a 3d array of shape (width,height,channels) or 2d array of shape (width,height)
    #and returns a 3d tensor of shape (channels,width,height)
    if len(img.shape) == 2:
        img = img[...,None]
    if len(img.shape) == 3 and img.shape[-1] == 1:
        img = np.repeat(img,3,axis=-1)
    if len(img.shape) == 3:
        return torch.tensor(img[:,:,:3]).permute(2,0,1)
    else:
        raise ValueError('img has wrong shape!')

def compute_covar(img, kernel_size = 7):
    
    mini_patches = None        
    for i in range(kernel_size//2,img.shape[0]-kernel_size//2):
        for j in range(kernel_size//2,img.shape[1]-kernel_size//2):
            mini_patch = img[i-(kernel_size//2):i+1+(kernel_size//2),j-(kernel_size//2):j+1+(kernel_size//2)]
            mini_patch = np.expand_dims(mini_patch, axis=-1)

            if i == kernel_size//2 and j == kernel_size//2:
                mini_patches = mini_patch
            else:
                mini_patches = np.concatenate((mini_patches,mini_patch),axis=-1)
                
    covar = np.empty((kernel_size,kernel_size))
    n = mini_patches.shape[2]
    for k in range(covar.shape[0]):
        for l in range(covar.shape[1]):
            covar[k,l] = ((mini_patches[k,l]-np.mean(mini_patches[k,l])).T)@(mini_patches[3,3]-np.mean(mini_patches[3,3]))/(n-1)
    
    return covar

def local_stats(out_img, in_img, grain_size):
    #Compute mean, variance and covariance and compare to Newson's method
    tmp_mdist = []
    tmp_vdist = []
    tmp_cdist = []
    for k in range(64):
        xidx = k//8
        yidx = k%8
        
        out_patch = out_img[64*xidx:64*(xidx+1), 64*yidx: 64*(yidx+1)]
        in_patch = in_img[64*xidx:64*(xidx+1), 64*yidx: 64*(yidx+1)]
        
        out_patch = out_patch[2:-2,2:-2] #Avoid border artefatcs
        in_patch = in_patch[2:-2,2:-2]
        
        mean_out = np.mean(out_patch)
        mean_in = np.mean(in_patch)
        tmp_mdist.append(np.abs(mean_in-mean_out))
        
        n = 60**2
        grain_size = round(grain_size*1000)/1000 #rounded for file names 
        if os.path.exists('./data/var/var'+str(k)+'_gs'+str(grain_size)+'.npy'):
            var_true = np.load('./data/var/var'+str(k)+'_gs'+str(grain_size)+'.npy')
        else:
            var_true = grain_variance(in_patch[0,0], grain_size)
            np.save('./data/var/var'+str(k)+'_gs'+str(grain_size)+'.npy', var_true)
        var_out = np.var(out_patch)*n/(n-1)
        tmp_vdist.append(np.abs(var_true-var_out))
        
        kernel_size = 7
        grain_size = round(grain_size*1000)/1000 #rounded for file names 
        if os.path.exists('./data/covar/covar'+str(k)+'_gs'+str(grain_size)+'.npy'):
            covar_true = np.load('./data/covar/covar'+str(k)+'_gs'+str(grain_size)+'.npy')
        else:
            covar_true = np.empty((kernel_size,kernel_size))
            for i in range(covar_true.shape[0]):
                for j in range(covar_true.shape[1]):
                    covar_true[i,j] = grain_covariance(in_patch[0,0], np.array([i-kernel_size//2,j-kernel_size//2]), grain_size)
                    
            np.save('./data/covar/covar'+str(k)+'_gs'+str(grain_size)+'.npy', covar_true)

        covar_out = compute_covar(out_patch, kernel_size = kernel_size)
        
        tmp_cdist.append(np.sum((covar_true-covar_out)**2))
    
    return tmp_mdist, tmp_vdist, tmp_cdist

def get_images(net, in_img, nat_path, idx, gs, tmp_idx):
    newson = ut.Newson()
    nat_img = iio.imread(nat_path)/255.
    if len(nat_img.shape)==3:
        nat_img = nat_img[:,:,0:1]
    else:
        nat_img = np.expand_dims(nat_img,-1)
    seed = np.random.randint(1e9)

    if not os.path.exists('./data/tmp_'+str(tmp_idx)+'/'):
        os.mkdir('./data/tmp_'+str(tmp_idx)+'/')

    newson('./data/1.png', './data/tmp_'+str(tmp_idx)+'/1_test_'+str(idx+1)+'.png', gs, seed)
    newson(nat_path, './data/tmp_'+str(tmp_idx)+'/nat_'+str(idx+1)+'.png', gs, seed)
    ref_1d = iio.imread('./data/tmp_'+str(tmp_idx)+'/1_test_'+str(idx+1)+'.png')[:,:,0]/255.
    natref_1d = iio.imread('./data/tmp_'+str(tmp_idx)+'/nat_'+str(idx+1)+'.png')[:,:,0]/255.
    
    out = net(torch.unsqueeze(torch.tensor(in_img).permute(2,0,1),0).float(), torch.tensor([gs]).float())[0]
    out_1d = torch2img(out.detach().cpu())[:,:,0]
    natout = net(torch.unsqueeze(torch.tensor(nat_img).permute(2,0,1),0).float(), torch.tensor([gs]).float())[0]
    natout_1d = torch2img(natout.detach().cpu())[:,:,0]
        
    return ref_1d, out_1d, natref_1d, natout_1d

class Metrics():
    def __init__(self, ssim, lpip, sifid, dists, gatys, jsd, image_folder):
        self.ssim = ssim
        self.lpips = lpip
        self.sifid = sifid
        self.dists = dists
        self.gatys = gatys
        self.jsd = jsd
        self.image_folder = image_folder

        
    def __call__(self, batchsize, grain_sizes, net):
        warnings.filterwarnings("ignore")
        tmp_idx = np.random.randint(1e6)
        while os.path.exists('./data/tmp_'+str(tmp_idx)+'/'):
            tmp_idx = np.random.randint(1e6)

        in_img = iio.imread('./data/1.png')[:,:,0:1]/255.
        
        dist_values = np.empty((len(grain_sizes), batchsize))
        distnat_values = np.empty((len(grain_sizes), batchsize))
        mean_values = np.empty((len(grain_sizes), batchsize))
        var_values = np.empty((len(grain_sizes), batchsize))
        covar_values = np.empty((len(grain_sizes), batchsize))
        ssim_values = np.empty((len(grain_sizes), batchsize))
        ssimnat_values = np.empty((len(grain_sizes), batchsize))
        lpips_values = np.empty((len(grain_sizes), batchsize))
        lpipsnat_values = np.empty((len(grain_sizes), batchsize))
        sifid_values = np.empty((len(grain_sizes), batchsize))
        sifidnat_values = np.empty((len(grain_sizes), batchsize))
        gatys_values = np.empty((len(grain_sizes), batchsize))
        gatysnat_values = np.empty((len(grain_sizes), batchsize))
        jsd_values = np.empty((len(grain_sizes), batchsize))
        jsdnat_values = np.empty((len(grain_sizes), batchsize))
        
        for idx, gs in enumerate(tqdm(grain_sizes)):
            for b_idx in range(batchsize):
                print('.',end='')
                sys.stdout.flush()
                
                nat_path = self.image_folder+'0'*(2-len(str(b_idx+1)))+str(b_idx+1)+'.png'
                ref_1d, out_1d, natref_1d, natout_1d = get_images(net, in_img, nat_path, idx, gs, tmp_idx)

                ref_3d = np.repeat(np.expand_dims(ref_1d,axis=-1),3,axis=-1)
                out_3d = np.repeat(np.expand_dims(out_1d,axis=-1),3,axis=-1)
                natref_3d = np.repeat(np.expand_dims(natref_1d,axis=-1),3,axis=-1)
                natout_3d = np.repeat(np.expand_dims(natout_1d,axis=-1),3,axis=-1)
                #MEAN, VAR & COVARIANCE
                a,b,c = local_stats(out_1d, in_img, gs)
                mean_values[idx,b_idx] = np.mean(a)
                var_values[idx,b_idx] = np.mean(b)
                covar_values[idx,b_idx] = np.mean(c)

                #DIST
                ref = torch.unsqueeze(torch.tensor(ref_3d).permute(2,0,1),dim=0)
                out = torch.unsqueeze(torch.tensor(out_3d).permute(2,0,1),dim=0)
                dist_values[idx,b_idx] = self.dists(ref.float(), out.float()).detach().item() 

                natref = torch.unsqueeze(torch.tensor(natref_3d).permute(2,0,1),dim=0)
                natout = torch.unsqueeze(torch.tensor(natout_3d).permute(2,0,1),dim=0)
                distnat_values[idx,b_idx] = self.dists(natref.float(), natout.float()).detach().item()

                #LPIPS
                lpips_values[idx,b_idx] = self.lpips(torch.unsqueeze(torch.tensor(ref_1d).float(),0), torch.unsqueeze(torch.tensor(out_1d).float(),0))
                lpipsnat_values[idx,b_idx] = self.lpips(torch.unsqueeze(torch.tensor(natref_1d).float(),0), torch.unsqueeze(torch.tensor(natout_1d).float(),0))

                #SSIM
                ssim_values[idx,b_idx] = self.ssim(ref_1d, out_1d)
                ssimnat_values[idx,b_idx] = self.ssim(natref_1d, natout_1d)

                #JSD
                jsd_values[idx,b_idx] = self.jsd(ref_1d, out_1d)
                jsdnat_values[idx,b_idx] = self.jsd(natref_1d, natout_1d)

                #SIFID
                os.mkdir('./sifid_ref_'+str(tmp_idx)+'/')
                os.mkdir('./sifid_out_'+str(tmp_idx)+'/')
                shutil.copyfile('./data/tmp_'+str(tmp_idx)+'/1_test_'+str(idx+1)+'.png','./sifid_ref_'+str(tmp_idx)+'/ref.png')
                iio.imwrite('./sifid_out_'+str(tmp_idx)+'/out.png', (255*out_1d).astype(np.uint8))
                sifid_values[idx,b_idx] = self.sifid('./sifid_ref_'+str(tmp_idx)+'/','./sifid_out_'+str(tmp_idx)+'/')
                    
                shutil.rmtree('./sifid_ref_'+str(tmp_idx)+'/')
                shutil.rmtree('./sifid_out_'+str(tmp_idx)+'/')

                #SIFID Natural
                os.mkdir('./sifid_ref_'+str(tmp_idx)+'/')
                os.mkdir('./sifid_out_'+str(tmp_idx)+'/')
                shutil.copyfile('./data/tmp_'+str(tmp_idx)+'/nat_'+str(idx+1)+'.png','./sifid_ref_'+str(tmp_idx)+'/ref.png')
                iio.imwrite('./sifid_out_'+str(tmp_idx)+'/out.png', (255*natout_1d).astype(np.uint8))
                sifidnat_values[idx,b_idx] = self.sifid('./sifid_ref_'+str(tmp_idx)+'/','./sifid_out_'+str(tmp_idx)+'/')
                    
                shutil.rmtree('./sifid_ref_'+str(tmp_idx)+'/')
                shutil.rmtree('./sifid_out_'+str(tmp_idx)+'/')

                #Gatys                
                gatys_values[idx,b_idx] = self.gatys((torch.tensor(ref_1d)[None,None,...]).float().cuda(), (torch.tensor(out_1d)[None,None,...]).float().cuda()).detach().item()
                gatysnat_values[idx,b_idx] = self.gatys((torch.tensor(natref_1d)[None,None,...]).float().cuda(), (torch.tensor(natout_1d)[None,None,...]).float().cuda()).detach().item()
            
            print(':')
            sys.stdout.flush()
        return mean_values, var_values, covar_values, dist_values, ssim_values, lpips_values, sifid_values, distnat_values, ssimnat_values, lpipsnat_values, sifidnat_values, gatys_values, gatysnat_values, jsd_values, jsdnat_values

def grain_variance(u, grain_size, sigma=0.8):
    
    def func(x):
        return np.exp(-((x/sigma)**2)/4) * cb(u,x,grain_size) *x
    
    integrale = scipy.integrate.quad(func,0,2*grain_size)[0]
    return integrale/(2*(sigma**2)) + (2**(-2*8))/12 #Second term is due to quantization

def grain_covariance(u, delta, grain_size, sigma=0.8):
    """
    delta is a 2d-array with respectively the displacement on x and y 
    """    
    def func(y,x):
        return np.exp(-(x**2 - 2*x*(np.cos(y)*delta[0]+np.sin(y)*delta[1]))/(4*(sigma)**2)) * cb(u,x,grain_size)*x
    
    integrale = scipy.integrate.dblquad(func, 0, 2*grain_size, -np.pi, np.pi)[0]
    return np.exp(-(delta[0]**2+delta[1]**2)/(4*(sigma)**2))*integrale/(4*np.pi*(sigma**2))
    
def gamma(grain_size, delta):
    #Compute the overlap area between two cirlces of same radius 'grain size' at distance 'delta'
    if delta == 0:
        # One circle totally overlaps the other.
        return np.pi * grain_size**2
    if delta >= 2*grain_size:
        # The circles don't overlap at all.
        return 0
    else:        
        alpha = np.arccos(delta / (2*grain_size))
        area = grain_size**2 * (2*alpha - np.sin(2*alpha))            
        return area

def cb(u, delta, grain_size):
    if delta >= 2*grain_size:
        # The circles don't overlap at all.
        return 0
    elif u == 1:
        return 0
    else:
        lamb = 1 / (np.pi * (grain_size**2))*np.log(1/(1-u))    
        cb_ = ((1-u)**2 )* (np.exp(lamb*gamma(grain_size,delta))-1)
        return cb_

def read_img(path):
    ''''takes a path and returns the image in (widht,height,3channels) format'''
    img = plt.imread(path)
    if len(img.shape) == 2:
        return np.repeat(img[...,None],3,axis=-1)
    elif len(img.shape)==3 and img.shape[-1]==1:
        return np.repeat(img,3,axis=-1)
    elif len(img.shape)==3 and img.shape[-1]!=1:
        return img[:,:,:3]
    else:
        raise ValueError('path gives input img with shape different from [widht;height] or [width;height;canals]')

print('functions: defined')
#%%
#-----------------------------------------------------------------------------#
#---------------------------# Compute metrics  #------------------------------#
#-----------------------------------------------------------------------------#


start_time = time.time()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

ssim = SSIM()
lpip = LPIPS()
sifid = SIFID()
dists = DISTS()
jsd = JSD_NSS()
gatys = tr.GatysLoss(style_indices = [1,6,11,20], style_weights = [1/(2**n) for n in [0,1,2,3]], vgg_type = 'ImageNet', vgg_pool = 'max', mode = 'Gatys')
image_folder = './data/natural_images_plus/'#'./data/L20/'
batchsize = 60 #20
save_folder = ''#'L20/'

grain_list = np.round(np.linspace(0.025,0.8,32),3)#np.array([0.01, 0.025, 0.05, 0.075, 0.1])#       

networks = ['./GrainNet/pretrained/grainnet.pt']

metric = Metrics(ssim, lpip, sifid, dists, gatys, jsd, image_folder)

results = {'name':      [nt.split('/')[-1][:-3] for nt in networks],
           'mean':      np.empty((len(networks),len(grain_list),batchsize)),
           'var':       np.empty((len(networks),len(grain_list),batchsize)),
           'covar':     np.empty((len(networks),len(grain_list),batchsize)),
           'DISTS':     np.empty((len(networks),len(grain_list),batchsize)),
           'DISTSnat':  np.empty((len(networks),len(grain_list),batchsize)),
           'SSIM':      np.empty((len(networks),len(grain_list),batchsize)),
           'SSIMnat':   np.empty((len(networks),len(grain_list),batchsize)),
           'LPIPS':     np.empty((len(networks),len(grain_list),batchsize)),
           'LPIPSnat':  np.empty((len(networks),len(grain_list),batchsize)),
           'SIFID':     np.empty((len(networks),len(grain_list),batchsize)),
           'SIFIDnat':  np.empty((len(networks),len(grain_list),batchsize)),
           'Gatys':     np.empty((len(networks),len(grain_list),batchsize)),
           'Gatysnat':  np.empty((len(networks),len(grain_list),batchsize)),
           'JSD-NSS':   np.empty((len(networks),len(grain_list),batchsize)),
           'JSD-NSSnat':np.empty((len(networks),len(grain_list),batchsize))}


for idx,net_name in enumerate(networks):
    state_dict = torch.load('./models/'+net_name)
    
    try:
        net = nets.GrainNet(block_nb = 3)
        net.load_state_dict(state_dict)
    except:
        try:
            net = nets.GrainNet(block_nb = 2)
            net.load_state_dict(state_dict)
        except:
            try:
                net = nets.GrainNet(block_nb = 1)
                net.load_state_dict(state_dict)
            except:
                raise ValueError('WRONG network path!')
    net.eval()

    a,b,c,d,e,f,g,h,i,j,k,l,m,n,o = metric(batchsize, grain_list, net)
    results['mean'][idx] = a
    results['var'][idx] = b
    results['covar'][idx] = c
    results['DISTS'][idx] = d
    results['SSIM'][idx] = e
    results['LPIPS'][idx] = f
    results['SIFID'][idx] = g
    results['DISTSnat'][idx] = h
    results['SSIMnat'][idx] = i
    results['LPIPSnat'][idx] = j
    results['SIFIDnat'][idx] = k
    results['Gatys'][idx] = l
    results['Gatysnat'][idx] = m
    results['JSD-NSS'][idx] = n
    results['JSD-NSSnat'][idx] = o
    print('One model done')
    sys.stdout.flush()

save_name = ''.join([nt.split('/')[-1][:-3] for nt in networks])
with open('./metrics/scores_'+save_name+'.pkl', 'wb') as f:
    pickle.dump(results, f)


#%%
#-----------------------------------------------------------------------------#
#-------------------------# Merge metrics files  #----------------------------#
#-----------------------------------------------------------------------------#
"""
files = ['./metrics/scoresA.pkl',
        './metrics/scoresB.pkl',
        './metrics/scoresC.pkl']

dicts = []
for fname in files:
    with open(fname, 'rb') as f:
        dicts.append(pickle.load(f))

new_dict = {key:np.concatenate([d[key] for d in dicts]) for key in dicts[0].keys()}

with open('./metrics/scores_total.pkl', 'wb') as f:
    pickle.dump(new_dict, f)
"""
#%%
#-----------------------------------------------------------------------------#
#-------------------------# Read n Rank Results  #----------------------------#
#-----------------------------------------------------------------------------#
"""
with open('./metrics/scores_total.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

#Raw scores
lines = loaded_dict['name']
columns = list(loaded_dict.keys())[1:]

values = []
for col_name in columns:
    values.append(np.mean(loaded_dict[col_name], axis=1)) 
    # for 4 first values: values.append(np.mean(loaded_dict[col_name][:,:4], axis=1))

raw_scores = pd.DataFrame(data=dict(zip(columns, [np.mean(v,axis=1) for v in values])), index = lines)
raw_scores_std = pd.DataFrame(data=dict(zip(columns, [np.std(v,axis=1) for v in values])), index = lines)

scores_by_grainsize = []
scores_by_grainsize_std = []

for i in range(loaded_dict['mean'].shape[1]):
    values = []
    for col_name in columns:
        values.append(loaded_dict[col_name][:,i])

    scores_by_grainsize.append(pd.DataFrame(data=dict(zip(columns, [np.mean(v,axis=1) for v in values])), index = lines))
    scores_by_grainsize_std.append(pd.DataFrame(data=dict(zip(columns, [np.std(v,axis=1) for v in values])), index = lines))

#------------------------------------------------------------------------#

raw_ranks = raw_scores.copy()
for col in columns:
    if (col != 'SSIM') and (col != 'SSIMnat'):
        raw_ranks[col] = np.argsort(np.argsort(raw_scores[col]))
    else:
        raw_ranks[col] = np.argsort(np.argsort(-raw_scores[col]))

raw_ranks['mean rank'] = np.mean(raw_ranks.to_numpy(), axis=1)
raw_ranks.sort_values('mean rank')

ranks_by_grainsize = []
for i in range(loaded_dict['mean'].shape[1]):
    rank = scores_by_grainsize[i].copy()
    for col in columns:
        if (col != 'SSIM') and (col != 'SSIMnat'):
            rank[col] = np.argsort(np.argsort(scores_by_grainsize[i][col]))
        else:
            rank[col] = np.argsort(np.argsort(-scores_by_grainsize[i][col]))

    rank['mean rank'] = np.mean(rank.to_numpy(), axis=1)
    rank.sort_values('mean rank')
    ranks_by_grainsize.append(rank)
"""
#%%
#-----------------------------------------------------------------------------#
#---------------------------# Visualize Results  #----------------------------#
#-----------------------------------------------------------------------------#

"""
shp = scores_by_grainsize[0].shape

for k in range(shp[1]):
    names = list(scores_by_grainsize[0].index)
    col = list(scores_by_grainsize[0].columns.values)[k]
    metrics = [[] for _ in range(shp[0])]
    metrics_std =  [[] for _ in range(shp[0])]
    for j in range(shp[0]):
        for i in range(len(scores_by_grainsize)):
            metrics[j].append(scores_by_grainsize[i][col][j])
            metrics_std[j].append(scores_by_grainsize_std[i][col][j])

    if col in ['mean','var','covar','Gatysnat','SSIMnat','LPIPSnat','DISTSnat','SIFIDnat','JSD-NSSnat']:
        plt.figure()
        for j in range(shp[0]):
            if names[j] in ['grainnet']:
                name = names[j]
                if name == 'grainnet': name = 'Ours'
                            
                plt.plot(np.linspace(0.025,0.8,32),np.array(metrics[j]),label=name)
                if col in ['mean','var','covar']:
                    plt.fill_between(np.linspace(0.025,0.8,32), np.array(metrics[j])-np.array(metrics_std[j]), np.array(metrics[j])+np.array(metrics_std[j]), alpha = 0.2)
                else:
                    plt.fill_between(np.linspace(0.025,0.8,32), np.array(metrics[j])-np.array(metrics_std[j])/60**0.5, np.array(metrics[j])+np.array(metrics_std[j])/60**0.5, alpha = 0.2)

        if col in ['mean','var','covar']:
            plt.title(col+' comparison')
        else:
            plt.title(col[:-3]+' comparison')
        plt.xlabel('grain size')
        plt.yscale('log')
        plt.legend()
        #plt.savefig('path/to/save/plot/'+col+'.png')
        plt.show()
"""

# %%
