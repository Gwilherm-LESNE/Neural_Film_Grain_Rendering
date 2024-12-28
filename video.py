import torch
import os
import numpy as np
import imageio.v2 as iio
import net as nets
import torchvision
import matplotlib.pyplot as plt

#%%
def torch2img(tensor):
    #Takes as entry a 4d tensor of shape (bsize,channels,width,height) 
    #and returns a 3d array of shape (width,height,channels)
    if len(tensor.shape) == 4:
        return tensor[0].permute(1,2,0).numpy()
    elif len(tensor.shape) == 3:
        return tensor.permute(1,2,0).numpy()
    else:
        raise ValueError('tensor has wrong shape!')

class VideoReader():
    """
    Reads all images from a folder
    """
    def __init__(self, folder):
        self.folder = folder
        if self.folder[-1] == '/':
            self.folder = self.folder[:-1]
        self.imgs = sorted(os.listdir(folder))
 
    def __call__(self,i):
        if i >= len(self.imgs):
            raise ValueError('Index out of bounds')

        return self.folder+'/'+self.imgs[i]
    
class Grainer():
    def __init__(self, vr, model_path, save_folder = None):
        self.vr = vr

        im = iio.imread(self.vr(0))
        if im.shape[2]>1:
            self.color = 1
        else:
            self.color = 0
        
        if save_folder is None:
            tmp = (self.vr.folder).split('/')
            self.save_folder = '/'.join(tmp[:-1])+'/'+tmp[-1]+'_'+self.method
        else:
            self.save_folder = save_folder

        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.net = nets.GrainNet(layer_nb = 2)
        self.net.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        self.net.eval()

    def grain(self, idx, gs):
        im = torch.unsqueeze(torchvision.io.read_image(self.vr(idx))/255.,0)
        if self.color:
            r = self.net(im[:,0:1,:,:],torch.tensor([gs])).detach().cpu()
            g = self.net(im[:,1:2,:,:],torch.tensor([gs])).detach().cpu()
            b = self.net(im[:,2:3,:,:],torch.tensor([gs])).detach().cpu()
            out = np.concatenate((torch2img(r),torch2img(g),torch2img(b)),axis=2)                
        else:
            out = torch2img(self.net(im,torch.tensor([gs])).detach().cpu())
        plt.imsave(self.save_folder+'/'+'0'*(6-len(str(idx)))+str(idx)+'.png',out)

    def __call__(self, gs):
        for i in range(len(self.vr.imgs)):
            print('.',end='')
            self.grain(i, gs)
        print('Done for '+self.method)


#vr = VideoReader()
#my_grainer = Grainer(vr, model_path = './models/GrainNet/grainnet.pt')
#my_grainer(0.1)

# %%
