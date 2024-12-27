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

import net as nets

print('Imports: done')

#%%

class GrainDataset(Dataset):
    def __init__(self,
                 label_folder,
                 max_h = 320,
                 max_w = 320):
        
        self.label_folder = label_folder
        
        self.max_w = max_w
        self.max_h = max_h
        
        hyper_params = pd.read_json(self.label_folder + 'hyperparameters.json', orient='table').values
        self.img_names = hyper_params[:,0]
        self.radiuses = hyper_params[:,2]
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, i, crop = True):
        grainy_img = torchvision.io.read_image(self.label_folder + (6-len(str(i)))*'0'+str(i)+'.png')[:1,1:,1:] 
        #change data range to [0,1] if it's not the case by default
        if grainy_img.dtype is torch.uint8:
            grainy_img = grainy_img.float()/255.

        if crop:
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

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        
    def forward(self, out, label):
        return nn.MSELoss()(out,label)

def train(parameters, label_folder, save_folder='/folder/'):
    
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
                                    "gamma": params[4]})
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
        

        if label_folder.__class__ == list:
            raise ValueError("'label_folder' argument in function 'train' is a list. You just need to put the path to grainy images.")

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
                }, './models'+save_folder+str(idx_name)+'/chkpt.pt')
        
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
          gamma = [0.99,1]
    )
    
    import time
    start_time = time.time()
    
    train(parameters,
          'path/to/your/dataset/folder',
          save_folder = '/Classifier/')
    
    end_time = time.time()
    
    print("Total training time elapsed:", end_time-start_time)

# %%
