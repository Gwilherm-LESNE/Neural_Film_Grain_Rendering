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
import argparse
import net as nets

print('Imports: done')

#%%

class GrainDataset(Dataset):
    def __init__(self,
                 label_folder,
                 max_h = 320,
                 max_w = 320):
        """
        Initialize the GrainDataset class
        label_folder: str 
            Path where to read grainy images and look for hyperparameters.json file
        max_h: int
            Maximum height of the images taken as input
        max_w: int
            Maximum width of the images taken as input
        """
        
        self.label_folder = label_folder
        
        self.max_w = max_w
        self.max_h = max_h
        
        hyper_params = pd.read_json(self.label_folder + 'hyperparameters.json', orient='table').values
        self.img_names = hyper_params[:,0]
        self.radiuses = hyper_params[:,2]
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, i):
        """
        Get data from the dataset based on its index
        i: int
            Data index

        returns
            grainy_img: grainy_image
            radius: grain_size
        """
        grainy_img = torchvision.io.read_image(self.label_folder + (6-len(str(i)))*'0'+str(i)+'.png')[:1,1:,1:] 
        #change data range to [0,1] if it's not the case by default
        if grainy_img.dtype is torch.uint8:
            grainy_img = grainy_img.float()/255.

        x_possible_coords = np.arange(0,grainy_img.shape[1]-self.max_h+1)
        y_possible_coords = np.arange(0,grainy_img.shape[2]-self.max_w+1)

        x = np.random.choice(x_possible_coords)
        y = np.random.choice(y_possible_coords)
        grainy_img = grainy_img[:,x:x+self.max_h,y:y+self.max_w]

        radius = torch.tensor([self.radiuses[i]])

        return grainy_img, radius
    
def plot_results(inputs, outputs, targets):
    plt.ioff()
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize= (10,10))
    idx = 0
    for row in ax:
        for col in row:
            im = inputs[idx].permute(1,2,0).cpu().numpy()
            col.imshow(im, cmap='gray', vmin=0, vmax=1)
            col.set_title("real size: "+str(targets[idx].item())[:5]+", estimated: "+str(outputs[idx].item())[:5])
            idx+=1
    return fig

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        
    def forward(self, out, label):
        return nn.MSELoss()(out,label)

def train(parameters, label_folder, save_folder):
    
    param_values = [v for v in parameters.values()]
    print('Parameter values are :' +str(param_values))
           
    if not os.path.exists('./models'+save_folder):
            os.mkdir(os.path.join('./models',save_folder))
    
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
        while os.path.exists(os.path.join('./models',save_folder,str(idx_name))):
            if os.path.exists(os.path.join('./models',save_folder,str(idx_name),'chkpt.pt')):
                hyper_params = dict(zip(parameters.keys(),params))
                checkpoint = torch.load(os.path.join('./models',save_folder,str(idx_name),'chkpt.pt'))
                chkpt_params = dict(zip(list(hyper_params.keys()), [checkpoint[key] for key in hyper_params.keys()]))
                
                if (chkpt_params == hyper_params) and (not checkpoint['training_finished']):
                    from_chkpt = True
                    break
                else:
                    idx_name+=1
            else:
                idx_name += 1
        if not from_chkpt:
            os.mkdir(os.path.join('./models',save_folder,str(idx_name)))
        writer = SummaryWriter(os.path.join('./models',save_folder,str(idx_name)))
        
        train_dataset = GrainDataset(label_folder = os.path.join(label_folder,'train/'))
        test_dataset = GrainDataset(label_folder = os.path.join(label_folder,'test/'))

        train_dataloader = DataLoader(train_dataset, batch_size = params[1], shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size = params[1], shuffle=True, drop_last=True)

        net = nets.Classifier()
        
        loss = nn.MSELoss()
        tloss = nn.MSELoss()

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
                inputs, radius = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize  
    
                outputs = net(inputs.float())

                loss_value = loss(outputs,radius)
                
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
                test_steps = 0
                net.eval()
                
                for i, data in enumerate(test_dataloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, radius= data[0].to(device), data[1].to(device)
                    
                    # forward + backward + optimize
     
                    outputs = net(inputs.float())
                    
                    loss_value = tloss(outputs, radius)
                        
                    test_loss += loss_value.item()

                    test_steps+=1
                                        
                
                print(f"Test Loss for epoch n°{epoch + 1}: {float(test_loss/test_steps):.2f}")
                sys.stdout.flush()
                
                writer.add_scalar("Loss/TestLoss", test_loss/test_steps, epoch)
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
                torch.save(state|dict(zip(parameters.keys(),params)), os.path.join('./models',save_folder,str(idx_name),'chkpt.pt'))
        
        #-------#Plot outputs#-------#
        inputs, radius = next(iter(test_dataloader))
        while inputs.shape[0] < 4:
            tmp_i, tmp_r = next(iter(test_dataloader))
            inputs = torch.cat((inputs,tmp_i),dim=0)
            radius = torch.cat((radius,tmp_r),dim=0)
        else:
            inputs = inputs[:4].to(device)
            radius = radius[:4].to(device)

        outputs = net(inputs.float()).detach()

        inputs = inputs.detach()
        radius = radius.detach()
        writer.add_figure('output vs. ground truth',
                        plot_results(inputs, outputs, radius),
                        global_step = epoch,
                        close=True)
        plt.close()
        #----------------------------#

        sys.stdout.write('Finished Training'+'\n')
        sys.stdout.flush()
        writer.add_hparams(
            {"lr": params[0], 
             "bsize": params[1], 
             "n_epoch": params[2],
             "step_size": params[3],
             "gamma": params[4]},
            {"training loss": train_loss/train_steps,
             "test loss": test_loss/test_steps,  
            },
        )
        
        save_name = os.path.join('./models',save_folder,str(idx_name),'fg'+str(idx_name)+'.pt')
    
        if not os.path.exists(os.path.join('./models',save_folder,str(idx_name))):
            os.mkdir(os.path.join('./models',save_folder,str(idx_name)))
        torch.save(net.state_dict(), save_name)
        
        
    writer.close()

    
print('Defining functions: done')
#%%

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i", '--input_path',metavar = 'input_path', help = "Path to the folder containing grainy images", type = str, required=True)

    parser.add_argument("-s", '--save_folder',metavar = 'save_folder', help = "Path where to save the model trained and checkpoints", type = str, default= 'Classifier/')
    parser.add_argument("-lr", '--learning_rate',metavar = 'learning_rate', help = "Learning_rate", type = float, default = 1e-03)
    parser.add_argument("-bs", '--batch_size', metavar = 'batch_size', help = 'batch_size', type = int, default = 16)
    parser.add_argument("-ne",'--n_epoch', metavar = 'n_epoch', help = "Number_of_epochs", type=int, default = 500)
    parser.add_argument("-ss", '--step_size', metavar = 'step_size', help = "Step_size for optimizer scheduler", type=int, default = 1)
    parser.add_argument("-ga", '--gamma', metavar = 'gamma', help = "Gamma for optimizer scheduler", type=float, default = 0.99)
    args = parser.parse_args()

    parameters = dict(lr = [args.learning_rate],
          batch_size = [args.batch_size],
          n_epoch = [args.n_epoch],
          step_size = [args.step_size],
          gamma = [args.gamma]
    )
    
    import time
    start_time = time.time()
    
    train(parameters,
          args.input_path,
          args.save_folder)
    
    end_time = time.time()
    
    print("Total training time elapsed:", end_time-start_time)

# %%
