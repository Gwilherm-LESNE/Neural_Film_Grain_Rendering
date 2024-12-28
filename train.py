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
import argparse

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
        self.radiuses = hyper_params[:,2]
        
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

        return ipt_img, label_img, radius
    
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
    def __init__(self, weight, style_indices = None, style_weights = None):
        super(StyleLoss, self).__init__()
                
        self.contentLoss = ContentLoss()
            
        self.textureLoss = TextureLoss(style_indices = style_indices, style_weights = style_weights)
        self.weight = weight
        
    def forward(self, out_img, label_img, test=False):
        texture = self.textureLoss(out_img, label_img)
        content = self.contentLoss(out_img,label_img)
        if test:
            return  texture+ self.weight * content, texture, content
        else:
            return  texture+ self.weight * content

class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        
        self.vgg19 = torchvision.models.vgg19(weights='IMAGENET1K_V1').features[:32].cuda()
        print("ImageNet's VGG weights for content loss successfully imported")
 
    def forward(self, out_img, label_img):
            
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
    
class TextureLoss(nn.Module):
    def __init__(self, style_indices = None, style_weights = None):
        super(TextureLoss, self).__init__()
        
        self.vgg19 = torchvision.models.vgg19(weights='IMAGENET1K_V1').features[:30].cuda()
        print("ImageNet's VGG weights successfully imported")
                     
        ####CHANGE MAXPOOL INTO AVERAGEPOOL####
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
            
            mean = torch.nn.MSELoss()(torch.mean(output_activations, axis=(2,3)),torch.mean(target_activations, axis=(2,3)))
            std = torch.nn.MSELoss()(torch.std(output_activations, axis=(2,3)),torch.std(target_activations, axis=(2,3)))
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
        torch.manual_seed(0)#Set seed
            
        torch.cuda.empty_cache() #Empty cache of the GPU

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sys.stdout.write('DEVICE TYPE :'+ device.type + '\n')
        sys.stdout.flush()

        ###Create a new SummaryWriter or resume at a checkpoint.
        idx_name=0
        from_chkpt = False
        while os.path.exists('./models'+save_folder+str(idx_name)+'/'):
            if os.path.exists('./models'+save_folder+str(idx_name)+'/chkpt.pt'):
                hyper_params = dict(zip(parameters.keys(),params))
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
        ###

        train_dataset = GrainDataset(ipt_folder = os.path.join(ipt_folder,'train/'), label_folder = os.path.join(label_folder,'train/'))
        test_dataset = GrainDataset(ipt_folder = os.path.join(ipt_folder,'test/'), label_folder = os.path.join(label_folder,'test/'))

        train_dataloader = DataLoader(train_dataset, batch_size = params[1], shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size = params[1], shuffle=True, drop_last=True)

        net = nets.GrainNet(activation=params[9], block_nb = params[12])

        loss = StyleLoss(weight = params[11], content_mode = params[10], style_indices=params[5], style_weights=params[6])
        tloss = StyleLoss(weight = params[11], content_mode = params[10], style_indices=params[5], style_weights=params[6])
            
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
                # get the inputs; data is a list of [inputs, labels, grain_size]
                inputs, labels, radius = data[0].to(device), data[1].to(device), data[2].to(device)

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
                    # get the inputs; data is a list of [inputs, labels, grain_size]
                    inputs, labels, radius = data[0].to(device), data[1].to(device), data[2].to(device)
                    
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

            if epoch%50==49:
                #Save checkpoints
                if epoch == n_epoch-1:
                    training_finished = True
                else:
                    training_finished = False
                
                state = dict({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'training_finished': training_finished})
                torch.save(state|dict(zip(parameters.keys(),params)), './models'+save_folder+str(idx_name)+'/chkpt.pt')
        
        inputs, targets, radius = next(iter(test_dataloader))
        if inputs.shape[0] < 4: #BATCH SIZE MUST BE AT LEAST 2!
            tmp_i, tmp_t, tmp_r = next(iter(test_dataloader))
            inputs = torch.cat((inputs,tmp_i),dim=0)
            targets = torch.cat((targets,tmp_t),dim=0)
            radius = torch.cat((radius,tmp_r),dim=0)
        else:
            inputs = inputs[:4].to(device)
            targets = targets[:4].to(device)
            radius = radius[:4].to(device)

        outputs = net(inputs.float(), radius.float()).detach()

        inputs = inputs.detach()
        targets = targets.detach()
        radius = radius.detach()
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
             "block_nb": params[12]},
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

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i", '--input_path',metavar = 'input_path', help = "Path to the folder containing grain-free images", type = str, required=True)
    parser.add_argument("-g", '--grain_path',metavar = 'grain_path', help = "Path to the folder containing grainy images", type = str, required=True)
    parser.add_argument("-s", '--save_path',metavar = 'save_path', help = "Path where to save the model trained and checkpoints", type = str, required=True)

    parser.add_argument("-lr", '--learning_rate',metavar = 'learning_rate', help = "Learning_rate", type = float, default = 1e-03)
    parser.add_argument("-bs", '--batch_size', metavar = 'batch_size', help = 'batch_size', type = int, default = 16)
    parser.add_argument("-ne",'--n_epoch', metavar = 'n_epoch', help = "Number_of_epochs", type=int, default = 500)
    parser.add_argument("-ss", '--step_size', metavar = 'step_size', help = "Step_size for optimizer scheduler", type=int, default = 1)
    parser.add_argument("-ga", '--gamma', metavar = 'gamma', help = "Gamma for optimizer scheduler", type=float, default = 0.99)
    parser.add_argument("-si", '--style_indices', nargs='+', metavar = 'style_indices', help = "Indices where to take the activation map of VGG", default = [1,6,11,20])
    parser.add_argument("-sw", '--style_weights', nargs='+', metavar = 'style_weights', help = "Weights for the corresponding activation maps of VGG", default = [1/(2**n) for n in [0,1,2,3]])
    parser.add_argument("-ac", '--activation', metavar = 'activation', help = "Activation function to use", type=str, default = 'tanh')
    parser.add_argument("-w", '--weight', metavar = 'weight', help = "Weight between content and style terms fro the loss function", type=float,  default = 1.)
    parser.add_argument("-bn", '--block_number', metavar = 'block_number', help = "Number of conv layer blocks in the architecture", type=int,  default = 2)
    args = parser.parse_args()

    if not args.block_number in [1,2,3]:
        raise ValueError("-ln (--block_number) provided must be 1,2 or 3")
    
    if not args.activation in ['tanh','sigmoid']:
        raise ValueError("-ac (--activation) provided must be 'tanh' or 'sigmoid'")

    parameters = dict(lr = [args.learning_rate],
          batch_size = [args.batch_size],
          n_epoch = [args.n_epoch],
          step_size = [args.step_size],
          gamma = [args.gamma],
          style_indices = [args.style_indices],
          style_weights = [args.style_weights],
          activ = [args.activation],
          weight = [args.weight],
          block_nb = [args.block_number]
    )

    start_time = time.time()
    
    train(parameters,
          args.input_path,
          args.grain_path,
          save_folder = args.save_path)
    
    end_time = time.time()
    
    print("Total training time elapsed:", end_time-start_time)

# %%
