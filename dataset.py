#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import numpy as np
import os
from PIL import Image
import subprocess
import pandas as pd
import warnings

#%%

class SynthethicDatasetGenerator():
    """
    This class creates a folder of synthethic 512x512 images (gradient, random shapes and different gray values).
    """
    
    def __init__(self, train_size = None, test_size = None):
        
        if train_size is None:
            self.train_size = 1e3
        else:
            self.train_size = train_size
        
        if test_size is None:
            self.test_size = 2e2
        else:
            self.test_size = test_size
        
    def create_dataset(self, save_folder):
        
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        train_folder = os.path.join(save_folder, 'train/')
        if not os.path.exists(train_folder):
            os.mkdir(train_folder)
        test_folder = os.path.join(save_folder, 'test/')
        if not os.path.exists(test_folder):
            os.mkdir(test_folder)
        
        print('Saving train dataset')
        for n in range(int(self.train_size)):
            image = self.create_image()
            img = Image.fromarray(np.uint8(255*image))
            fname = (6-len(str(n)))*'0'+str(n)
            img.save(train_folder+fname+'.png')
            if n%10 == 0:
                print('.',end='')
            if n%100 == 0:
                print('Train image n° ',n,' saved')
                
        print('Saving test dataset')
        for n in range(int(self.test_size)):
            image = self.create_image()
            img = Image.fromarray(np.uint8(255*image))
            fname = (6-len(str(n)))*'0'+str(n)
            img.save(test_folder+fname+'.png')
            if n%10 == 0:
                print('.',end='')
            if n%100 == 0:
                print('Test image n° ',n,' saved')
            
    def create_image(self):
        img = np.ones((512,512))
        
        bckgrd_type = np.random.randint(2)#0 for constant, 1 for gradient
        if bckgrd_type == 0:
            img *= np.random.randint(256)/256. #Value of the constant
        else:
            bckgrd_subtype = np.random.randint(4) #Orientation of the gradient
            for i in range(img.shape[0]):
                if bckgrd_subtype == 0:
                    img[:,i] = (i+1)/img.shape[0]
                elif bckgrd_subtype == 1:
                    img[:,i] = 1 - (i+1)/img.shape[0]
                elif bckgrd_subtype == 2:
                    img[i,:] = (i+1)/img.shape[0]
                else:
                    img[i,:] = 1 - (i+1)/img.shape[0]
        
        shape = np.random.randint(2) #Telling which shape to draw.
        N = np.random.randint(20) #number of shape
        if shape == 0: #Rectangles
            for s in range(N):
                x = np.random.randint(img.shape[0])
                y = np.random.randint(img.shape[1])
                dx = max(min(np.random.choice([-1,1])*np.random.randint(16,img.shape[0]//2),img.shape[0]-x),-x)
                dy = max(min(np.random.choice([-1,1])*np.random.randint(16,img.shape[1]//2),img.shape[1]-y),-y)
                v = np.random.randint(256)/256.
                
                img[min(x,x+dx):max(x,x+dx),min(y,y+dy):max(y,y+dy)] = v
            return img
        else: #Disks
            x = np.random.randint(img.shape[0], size=N)
            y = np.random.randint(img.shape[1], size=N)
            r = np.random.randint(16, img.shape[0]//4, size=N)
            v = np.random.randint(256, size=N)/256.             
                
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    for idx in range(N):
                        if (i-x[idx])**2+(j-y[idx])**2 < r[idx]**2:
                            img[i,j] = v[idx]       
            return img
            
    
class GrainDatasetGenerator():
    """
    This class creates a folder of grainy images based on a folder of gray images 
    which path is given as an argument.
    images must end with '.png'
    """
    def __init__(self, root, newson_path, grain_clean_ratio = 20):
        
        #Indicates how many grainy images are created for 1 clean image
        self.grain_clean_ratio = grain_clean_ratio 
        
        self.img_names = []
        self.seed_table = []
        self.radius_table = []
        self.zoom_table = []

        self.newson_path = newson_path #path to the executable file of Newson etal. code (bin/film_grain_rendering_main)
        self.radius_range = np.linspace(0.025,0.8,32)
        self.NMonteCarlo = 1e5
        self.zoom_range = [1.]
        self.grainSigma = 0.
        self.filterSigma = 0.8
        self.algorithmID = 0.
        
        self.root = root
        if self.root[-1] != '/':
            self.root += '/'
            
        self.fnames = []
        for r, d, f in os.walk(root):
            for file in f:
                if file.endswith(".png"):
                    self.fnames.append(file)
                    
    def __call__(self, save_folder = None):
        if save_folder is None:
            new_path = ''
            old_path = self.root[:-1].split('/')
            for idx in range(len(old_path)-1):
                new_path += old_path[idx] +'/'
                
            new_path += old_path[-1]+'_grain/'            
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            save_folder = new_path
        
        root_content = os.listdir(self.root)
        is_dir = np.array([os.path.isdir(self.root+el) for el in root_content])
        if sum(is_dir)>0:
            root_dirs = np.array(root_content)[is_dir]
            for dirit in root_dirs:
                if not os.path.exists(save_folder + dirit):
                    os.mkdir(save_folder + dirit)
                self.create(self.root + dirit + '/', save_folder + dirit + '/')
        else:
            self.create(self.root, save_folder)
            
    def create(self, src_img_folder, target_img_folder):
        fname_idx = 0
        for r, d, f in os.walk(src_img_folder):
            for img_name in f:
                if img_name.endswith(".png"):
                    for _ in range(self.grain_clean_ratio):
                        img_path = os.path.join(r, img_name)
                        save_name = (6-len(str(fname_idx)))*'0'+str(fname_idx) + '.png'
                        save_path = os.path.join(target_img_folder, save_name)
                        seed = np.random.randint(1000)
                        radius = np.random.choice(self.radius_range)
                        zoom = 1.          
                        subprocess.run([self.newson_path,
                                        img_path,
                                        save_path,
                                        "-r",str(radius),
                                        "-grainSigma",str(self.grainSigma),
                                        "-filterSigma",str(self.filterSigma),
                                        "-zoom",str(zoom),
                                        "-algorithmID",str(self.algorithmID),
                                        "-NmonteCarlo",str(self.NMonteCarlo),
                                        "-randomizeSeed",str(seed)])
                        
                        self.img_names.append(img_name)
                        self.seed_table.append(seed)
                        self.radius_table.append(radius)
                        self.zoom_table.append(zoom)
                        
                        fname_idx += 1
        
        json_file = {'img_name':self.img_names,'seed':self.seed_table,'radius':self.radius_table,'zoom':self.zoom_table}
        df = pd.DataFrame(data = json_file)
        df.to_json(target_img_folder+'hyperparameters.json', orient='table')
        self.img_names = []
        self.seed_table = []
        self.radius_table = []
        self.zoom_table = []

        
class RGB2Gray_Dataset_Convertor():
    """
    This class aims at converting a dataset of RGB images (.jpg or .png) into a gray '.png' one.
    Only png, jpg, bmp or tiff are handled.
    """
    def __init__(self, dataset_root):
        
        self.root = dataset_root
        if self.root[-1] != '/':
            self.root += '/'
        
    def __call__(self, save_folder = None):
        """
        Converts a dataset of RGB images (.jpg or .png) into a gray '.png' one. 
        This function can deal with self.root having images inside of it or directories but not with both.
        The goal is to handle both datasets subdivided into train/test/valid folders and 'raw' datasets.
                
        Parameters
        ----------
        save_folder : str, optional
            Indicates where to put the gray images in. If None, a folder next to root called 'root_gray' is created and taken. The default is None.

        Returns
        -------
        None.

        """
        if save_folder is None:
            new_path = ''
            old_path = self.root[:-1].split('/')
            for idx in range(len(old_path)-1):
                new_path += old_path[idx] +'/'
                
            new_path += old_path[-1]+'_gray/'            
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            save_folder = new_path
        
        root_content = os.listdir(self.root)
        is_dir = np.array([os.path.isdir(self.root+el) for el in root_content])
        if sum(is_dir)>0:
            root_dirs = np.array(root_content)[is_dir]
            for dirit in root_dirs:
                if not os.path.exists(save_folder + dirit):
                    os.mkdir(save_folder + dirit)
                self.convert(self.root + dirit + '/', save_folder + dirit + '/')
        else:
            self.convert(self.root, save_folder)
    
    def convert(self, src_img_folder, target_img_folder):
        idx_name = 0
        for r, d, f in os.walk(src_img_folder):
            for file in f:
                if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".tiff") or file.endswith(".bmp"):
                    img = Image.open(os.path.join(r, file)).convert('L')
                    fname = (6-len(str(idx_name)))*'0'+str(idx_name)
                    img.save(target_img_folder+fname+'.png')
                    idx_name += 1


class Newson():
    def __init__(self, path = None):
        if path is None:
            self.path = "Film_Grain_Rendering_GPU/bin/film_grain_rendering_main"
        else:
            self.path = path

    def __call__(self, input_path, save_path, grain_size, seed = 1, zoom = 1, it_nb = 1e5, verb = 0, color = 0):
        if seed == 0 and color == 1:
            seed = 1
            warnings.warn("seed is fixed and color is True, those two options aren't compatible, so we dropped the seed constraint")
        subprocess.run([self.path,
                        input_path,
                        save_path,
                        "-r", str(grain_size),
                        "-grainSigma",str(0.),
                        "-filterSigma",str(0.8),
                        "-zoom", str(zoom),
                        "-algorithmID",str(0.),
                        "-NmonteCarlo",str(it_nb),
                        "-randomizeSeed",str(seed),
                        "-verbose",str(verb),
                        "-color", str(color)])
    

# %%
