#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

import torchvision
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from itertools import product
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
import time
import net as nets

print('Imports: done')

#%%

class GrainDataset(Dataset):
    def __init__(self, ipt_folder, 
                 label_folder,
                 max_h = 320,
                 max_w = 320):
        
        self.ipt_folder = ipt_folder
        self.label_folder = label_folder
        
        self.max_w = max_w
        self.max_h = max_h
        
        hyper_params = pd.read_json(self.label_folder + 'hyperparameters.json', orient='table').values
        self.img_names = hyper_params[:,0]
        self.seeds = hyper_params[:,1]
        self.radiuses = hyper_params[:,2]
        self.zooms = hyper_params[:,3]
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, i, crop = True):
        ipt_img = torchvision.io.read_image(self.ipt_folder + self.img_names[i])[:1,1:,1:] #keep only one channel and drop first line and first column as they have artifacts
        label_img = torchvision.io.read_image(self.label_folder + (6-len(str(i)))*'0'+str(i)+'.png')[:1,1:,1:] 
        #change data range to [0,1] if it's not the case by default
        if ipt_img.dtype is torch.uint8:
            ipt_img = ipt_img.float()/255.
        if label_img.dtype is torch.uint8:
            label_img = label_img.float()/255.

        if crop:
            x_possible_coords = np.arange(0,ipt_img.shape[1]-self.max_h+1)
            y_possible_coords = np.arange(0,ipt_img.shape[2]-self.max_w+1)

            x = np.random.choice(x_possible_coords)
            y = np.random.choice(y_possible_coords)
            ipt_img = ipt_img[:,x:x+self.max_h,y:y+self.max_w]
            label_img = label_img[:,x:x+self.max_h,y:y+self.max_w]

        radius = torch.tensor([self.radiuses[i]])
        zoom = torch.tensor([self.zooms[i]])
        seed = torch.tensor([self.seeds[i]])

        return ipt_img, label_img, radius, zoom, seed
    
def plot_results(outputs, targets):
    plt.ioff()
    fig = plt.figure(figsize=(30, 60))
    for idx in np.arange(4):
        im1 = targets[idx].permute(1,2,0).cpu().numpy()
        im2 = outputs[idx].permute(1,2,0).cpu().numpy()
        
        ax = fig.add_subplot(2, 4, 2*idx+1, xticks=[], yticks=[])
        ax.imshow(im1, cmap='gray', vmin=0, vmax=1)
        ax.set_title("Grain n°"+str(idx)+", target")
        ax = fig.add_subplot(2, 4, 2*idx+2, xticks=[], yticks=[])
        ax.imshow(im2, cmap='gray', vmin=0, vmax=1)
        ax.set_title("Grain n°"+str(idx)+", output")
    return fig

def load_run(path):
  event_acc = event_accumulator.EventAccumulator(path)
  event_acc.Reload()
  data = {}

  for tag in sorted(event_acc.Tags()["scalars"]):
    x, y = [], []

    for scalar_event in event_acc.Scalars(tag):
      x.append(scalar_event.step)
      y.append(scalar_event.value)

    data[tag] = (np.asarray(x), np.asarray(y))
  return data

class StyleLoss(nn.Module):
    def __init__(self, weight, content_mode = None, style_indices = None, style_weights = None, vgg_type = 'Gatys', vgg_pool = 'max', style_mode = 'Gatys'):
        super(StyleLoss, self).__init__()
        
        self.style_mode = style_mode
        if content_mode is None:
            self.contentLoss = lambda a,b: 0
        else:
            self.contentLoss = ContentLoss(mode=content_mode)
            
        self.gatysLoss = GatysLoss(style_indices = style_indices, style_weights = style_weights, vgg_type = vgg_type, vgg_pool = vgg_pool, mode = self.style_mode)
        self.weight = weight
        
    def forward(self, out_img, label_img, test=False):
        texture = self.gatysLoss(out_img, label_img)
        content = self.contentLoss(out_img,label_img)
        if test:
            return  texture+ self.weight * content, texture, content
        else:
            return  texture+ self.weight * content

class ContentLoss(nn.Module):
    def __init__(self, mode = 'Laplacian'):
        super(ContentLoss, self).__init__()
        
        self.mode = mode     
        
        if self.mode == 'Gatys':
            self.vgg19 = torchvision.models.vgg19(weights='IMAGENET1K_V1').features[:32].cuda()
            print("ImageNet's VGG weights for content loss successfully imported")
        
        elif self.mode == 'Laplacian':
            self.kernel = torch.tensor([[[[0,-1,0],
                                          [-1,4,-1],
                                          [0,-1,0]]]])
            self.kernel = self.kernel.float().cuda()
        
        elif self.mode == 'Gaussian':
            self.kernel = torch.tensor([[[[1,2,1],
                                          [2,4,2],
                                          [1,2,1]]]])/16
        
            self.kernel = self.kernel.float().cuda()
            self.kernel.requires_grad = False
        elif self.mode == 'L2':
            pass        
        else:
            raise ValueError("Content loss mode not handled!")
            
    def forward(self, out_img, label_img):
        if self.mode in ['Laplacian','Gaussian']:
            fltr_out_img = torch.nn.functional.conv2d(out_img, self.kernel, padding=1)
            fltr_label_img = torch.nn.functional.conv2d(label_img, self.kernel, padding=1)
            return torch.sum((fltr_out_img - fltr_label_img)**2)
        
        elif self.mode == 'Gatys':
            
            if out_img.shape[1] == 1: #For grayscale images
                out_img = out_img.repeat(1,3,1,1)
                    
            if label_img.shape[1] == 1: #For grayscale images
                label_img = label_img.repeat(1,3,1,1)
     
            preprocess = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                              std=[0.229, 0.224, 0.225]),])
            output_activ = self.vgg19(preprocess(out_img))
            label_activ = self.vgg19(preprocess(label_img))
            output_activ = output_activ/(output_activ.shape[2]*output_activ.shape[3])
            label_activ = label_activ/(label_activ.shape[2]*label_activ.shape[3])
            loss = torch.nn.MSELoss()(output_activ,label_activ)
                    
            return loss 
        
        elif self.mode == 'L2':
            loss = torch.nn.MSELoss()(out_img,label_img)
            return loss
        else:
            raise ValueError('mode not handled.')                      
    
class GatysLoss(nn.Module):
    def __init__(self, style_indices = None, style_weights = None, vgg_type = 'ImageNet', vgg_pool = 'max', mode = 'Gatys'):
        super(GatysLoss, self).__init__()
        
        self.mode = mode
        if vgg_type == 'Gatys':
            self.vgg19 = torchvision.models.vgg19().features.cuda()
            try:
                pretrained_dict = torch.load('./models/vgg.pth')
                for param, item in zip(self.vgg19.parameters(), pretrained_dict.keys()):
                    param.data = pretrained_dict[item].type(torch.FloatTensor).cuda()
                print("Gatys' VGG weights successfully imported")
            except:
                raise ValueError("'vgg.pth' file is not in the 'models' directory")
            self.vgg19 = self.vgg19[:30]
                        
        elif vgg_type == 'ImageNet':
            self.vgg19 = torchvision.models.vgg19(weights='IMAGENET1K_V1').features[:30].cuda()
            print("ImageNet's VGG weights successfully imported")
            
        else:
            raise ValueError(" wrong 'vgg_type' argument value. Only 'Gatys' or 'ImageNet' are handled.")
         
        ####CHANGE MAXPOOL INTO AVERAGEPOOL####
        if vgg_pool == 'avg':
            pool_indices = [4,9,18,27]
            for idx in pool_indices:
                self.vgg19[idx] = nn.AvgPool2d(kernel_size=2, stride=2)

         
        for param in self.vgg19.parameters():
            param.requires_grad = False
        self.vgg19.eval()
        
        
        
        if style_indices is None:
            self.style_indices = [1,6,11,20,29]
        else:
            self.style_indices = style_indices
            
        if style_weights is None:
            self.style_weights = torch.ones(len(self.style_indices))
        else:
            self.style_weights = style_weights
        
        if len(self.style_indices) != len(self.style_weights):
            raise ValueError("Arguments 'style_weights' and 'style_indices' don't have the same length, please check them.")
            
    def gram_matrix(self, activation_maps):
        flat_maps = torch.flatten(activation_maps, start_dim=2)
        
        (bsize, channel_nb, activation_size) = flat_maps.shape
        
        matrix = torch.bmm(flat_maps,flat_maps.transpose(1, 2))
        return matrix/(channel_nb*activation_size)
    
    def get_acti(self, img):
        if img.shape[1] == 1: #For grayscale images
            img = img.repeat(1,3,1,1)
        
        preprocess = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                      std=[0.229, 0.224, 0.225]),])
        gram_list = []
        for idx,style_idx in enumerate(self.style_indices):
            net = self.vgg19[:style_idx+1]
            
            activations = net(preprocess(img))
            gram_list.append(self.gram_matrix(activations)[0].detach().cpu().numpy())
            
        return gram_list
                    
    def forward(self,output,target):
        if output.shape[1] == 1: #For grayscale images
            output = output.repeat(1,3,1,1)
            
        if target.shape[1] == 1: #For grayscale images
            target = target.repeat(1,3,1,1)
        
        preprocess = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                      std=[0.229, 0.224, 0.225]),])
        loss = 0.
        for idx,style_idx in enumerate(self.style_indices):
            net = self.vgg19[:style_idx+1]
            
            output_activations = net(preprocess(output))
            target_activations = net(preprocess(target))
            
            if self.mode == 'Gatys':
                gram_output = self.gram_matrix(output_activations)
                gram_target = self.gram_matrix(target_activations)
                layer_loss = torch.nn.MSELoss()(gram_output,gram_target)
                
            elif self.mode == 'AdaIN':
                mean = torch.nn.MSELoss()(torch.mean(output_activations, axis=(2,3)),torch.mean(target_activations, axis=(2,3)))
                std = torch.nn.MSELoss()(torch.std(output_activations, axis=(2,3)),torch.std(target_activations, axis=(2,3)))
                layer_loss = mean + std

            elif self.mode == 'LocalAdaIN':
                kernel = torch.ones((9,9))/49
                krnl = kernel.view(1, 1, 9, 9).repeat(output_activations.shape[1], 1, 1, 1)
                if output_activations.is_cuda:
                    krnl = krnl.cuda()
                krnl.requires_grad = False
                
                out_mean = torch.nn.functional.conv2d(output_activations, krnl, padding='same', groups=output_activations.shape[1])
                out_std = torch.sqrt(torch.nn.functional.conv2d((output_activations-out_mean)**2, krnl, padding='same', groups=output_activations.shape[1]))
                tgt_mean = torch.nn.functional.conv2d(target_activations, krnl, padding='same', groups=output_activations.shape[1])
                tgt_std = torch.sqrt(torch.nn.functional.conv2d((target_activations-tgt_mean)**2, krnl, padding='same', groups=output_activations.shape[1]))

                mean = torch.nn.MSELoss()(out_mean,tgt_mean)
                std = torch.nn.MSELoss()(out_std,tgt_std)
                layer_loss = mean + std
                
            loss += self.style_weights[idx]*layer_loss
            
        return loss

def train(parameters, ipt_folder, label_folder, save_folder='/folder/'):
    
    param_values = [v for v in parameters.values()]
    print('Parameter values are :' +str(param_values))
    
    if not os.path.exists('./models'+save_folder):
        os.mkdir('./models'+save_folder)
    
    for run_id, params in enumerate(product(*param_values)):
        print('_'*50)
        print(' '*16+'RUN number ',run_id+1)
        print('_'*50)
        print(params)
        torch.manual_seed(0)
            
        torch.cuda.empty_cache() #empty cache of the GPU

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sys.stdout.write('DEVICE TYPE :'+ device.type + '\n')
        sys.stdout.flush()

        #Create a new SummaryWriter or resume at a checkpoint.
        idx_name=0
        from_chkpt = False
        while os.path.exists('./models'+save_folder+str(idx_name)+'/'):
            if os.path.exists('./models'+save_folder+str(idx_name)+'/chkpt.pt'):
                
                hyper_params = dict({"lr": params[0], 
                                    "bsize": params[1], 
                                    "n_epoch": params[2],
                                    "step_size": params[3],
                                    "gamma": params[4],
                                    "style_indices": params[5],
                                    "style_weights": params[6],
                                    "VGG_type": params[7],
                                    "VGG_pool": params[8],
                                    "activ": params[9],
                                    "content_mode": params[10],
                                    "content_weight": params[11],
                                    "style_mode": params[12],
                                    "layer_nb": params[13]})
                checkpoint = torch.load('./models'+save_folder+str(idx_name)+'/chkpt.pt')
                chkpt_params = dict(zip(list(hyper_params.keys()), [checkpoint[key] for key in hyper_params.keys()]))
                
                if (chkpt_params == hyper_params) and (not checkpoint['training_finished']):
                    from_chkpt = True
                    break
                else:
                    idx_name+=1
            else:
                idx_name += 1
        if not from_chkpt:
            os.mkdir('./models'+save_folder+str(idx_name))
        writer = SummaryWriter('./models'+save_folder+str(idx_name))
        
        train_dataset = GrainDataset(ipt_folder = os.path.join(ipt_folder,'train/'), label_folder = os.path.join(label_folder,'train/'))
        test_dataset = GrainDataset(ipt_folder = os.path.join(ipt_folder,'test/'), label_folder = os.path.join(label_folder,'test/'))

        train_dataloader = DataLoader(train_dataset, batch_size = params[1], shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size = params[1], shuffle=True, drop_last=True)

        net = nets.GrainNet(activation=params[9], layer_nb = params[14])

        loss = StyleLoss(weight = params[11], content_mode = params[10], style_indices=params[5], style_weights=params[6], vgg_type = params[7], vgg_pool = params[8], style_mode = params[12])
        tloss = StyleLoss(weight = params[11], content_mode = params[10], style_indices=params[5], style_weights=params[6], vgg_type = params[7], vgg_pool = params[8], style_mode = params[12])
            
        optimizer = optim.Adam(net.parameters(), lr=params[0])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params[3], gamma=params[4])
        
        net.to(device)
    
        ##### --------- #####
        
        sys.stdout.write('Starting Training'+'\n')
        sys.stdout.flush()

        n_epoch = params[2]
        start_epoch = 0

        if from_chkpt:
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']+1
        
        for epoch in range(start_epoch, n_epoch):  # loop over the dataset multiple times
            
            print('-'*20)
            print(' '*5+'Epoch n°'+str(epoch+1))
            print('-'*20)
            sys.stdout.flush()
            
            running_loss = 0.0
            train_loss = 0.0
            train_steps = 0
            net.train(True)
            
            for i, data in enumerate(train_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels, radius, zoom = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize  
                outputs = net(inputs.float(), radius.float())

                loss_value = loss(outputs,labels)
                
                loss_value.backward()

                optimizer.step()
                
                # print statistics
                running_loss += loss_value.item()
                train_loss += loss_value.item()
                train_steps += 1
                
                if i % 10 == 9: # print every 10 mini-batches
                    print(f'[epoch n°{epoch + 1}, batch n°{i + 1:5d}] loss: {running_loss / 500:.3f}')
                    sys.stdout.flush()
                    running_loss = 0.0
            
            writer.add_scalar("Loss/train", train_loss/train_steps, epoch)
            writer.flush()
            
            scheduler.step()
            
            ##### Test part #####
            with torch.no_grad():
                
                test_loss = 0.0
                texture_loss = 0.
                content_loss = 0.
                test_steps = 0
                net.eval()
                
                for i, data in enumerate(test_dataloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels, radius, zoom = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
                    
                    # forward + backward + optimize   
                    outputs = net(inputs.float(), radius.float())
                    
                    loss_value, texture, content = tloss(outputs, labels, test=True)
                        
                    test_loss += loss_value.item()
                    texture_loss += texture.item()
                    content_loss += content.item()
                    test_steps+=1
                                        
                
                print(f"Test Loss for epoch n°{epoch + 1}: {float(test_loss/test_steps):.2f}")
                sys.stdout.flush()
                
                writer.add_scalar("Loss/TestLoss", test_loss/test_steps, epoch)
                writer.add_scalar("Loss/TextureLoss", texture_loss/test_steps, epoch)
                writer.add_scalar("Loss/ContentLoss", content_loss/test_steps, epoch)
                writer.flush()

            if epoch%20==19:
                #Save checkpoints
                if epoch == n_epoch-1:
                    training_finished = True
                else:
                    training_finished = False
                
                torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'training_finished': training_finished,
                'lr': params[0], 
                'bsize': params[1], 
                'n_epoch': params[2],
                'step_size': params[3],
                'gamma': params[4],
                'style_indices': params[5],
                'style_weights': params[6],
                'VGG_type': params[7],
                'VGG_pool': params[8],
                'activ': params[9],
                'content_mode': params[10],
                'content_weight': params[11],
                'style_mode': params[12],
                'layer_nb': params[13],
                }, './models'+save_folder+str(idx_name)+'/chkpt.pt')
        
        inputs, targets, radius, zoom, _ = next(iter(test_dataloader))
        if inputs.shape[0] < 4:
            tmp_i, tmp_t, tmp_r, tmp_z, _ = next(iter(test_dataloader))
            inputs = torch.cat((inputs,tmp_i),dim=0)
            targets = torch.cat((targets,tmp_t),dim=0)
            radius = torch.cat((radius,tmp_r),dim=0)
            zoom = torch.cat((zoom,tmp_z),dim=0)
        else:
            inputs = inputs[:4].to(device)
            targets = targets[:4].to(device)
            radius = radius[:4].to(device)
            zoom = zoom[:4].to(device)

        if params[13]=='SuperRes':
            outputs = net(inputs.float(), radius.float(), zoom.float()).detach()
        else:      
            outputs = net(inputs.float(), radius.float()).detach()

        inputs = inputs.detach()
        targets = targets.detach()
        radius = radius.detach()
        zoom = zoom.detach()
        writer.add_figure('output vs. ground truth',
                        plot_results(outputs, targets),
                        global_step = epoch,
                        close=True)
        plt.close()

        sys.stdout.write('Finished Training'+'\n')
        sys.stdout.flush()
        writer.add_hparams(
            {"lr": params[0], 
             "bsize": params[1], 
             "n_epoch": params[2],
             "step_size": params[3],
             "gamma": params[4],
             "style_indices": str(params[5]),
             "style_weights": str(params[6]),
             "VGG_type": params[7],
             "VGG_pool": params[8],
             "activ": params[9],
             "content_mode": params[10],
             "content_weight": params[11],
             "style_mode": params[12],
             "layer_nb": params[13]},
            {"training loss": train_loss/train_steps,
             "test loss": test_loss/test_steps,
             "texture loss": texture_loss/test_steps,   
             "content loss": content_loss/test_steps,   
            },
        )
        
        save_name = './models'+save_folder+str(idx_name)+'/fg'+str(idx_name)+'.pt'
    
        if not os.path.exists('./models'+save_folder+str(idx_name)):
            os.mkdir('./models'+save_folder+str(idx_name))
        torch.save(net.state_dict(), save_name)
        
        
    writer.close()

    
print('Defining functions: done')
#%%

if __name__=='__main__':

    parameters = dict(lr = [1e-03],
          batch_size = [16],
          n_epoch = [500],
          step_size = [1],
          gamma = [0.99],
          style_indices = [[1,6,11,20]],
          style_weights = [[1/(2**n) for n in [0,1,2,3]]],
          vgg_type = ['ImageNet'],#'Gatys',
          vgg_pool = ['max'],#'avg'
          activ = ['tanh'],#'sigmoid'
          content_mode = ['Gatys'],
          weight = [1],
          style_mode = ['AdaIN'],
          layer_nb = [1,2,3]
    )

    start_time = time.time()
    
    train(parameters,
          'path/to/input/clean/images',
          'path/to/label/grainy/images',
          save_folder = '/GrainNet/')
    
    end_time = time.time()
    
    print("Total training time elapsed:", end_time-start_time)

# %%
