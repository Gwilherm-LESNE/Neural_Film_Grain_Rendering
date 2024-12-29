#%%
import argparse
import torchvision
import torch
import net as nt
from datetime import datetime

#%%
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input',metavar = 'input', help = "Path to the input grain-free image", type = str, required=True)
    
    parser.add_argument("-o", '--output',metavar = 'output', help = "Path to save the output grainy image", type = str, default='output.png')
    parser.add_argument("-gs", '--grain_size',metavar = 'input', help = "Grain size to use", type = float, default = 0.1)
    parser.add_argument("-s", '--seed',metavar = 'seed', help = "Path to the input grain-free image", type = int, default=int(datetime.now().timestamp()))
    parser.add_argument("-m", '--model',metavar = 'model', help = "Path to the model to use", type = str, default='./models/GrainNet/default/grainnet.pt')
    parser.add_argument("-c", '--color',metavar = 'color', help = "Tells if the image is color or grayscale", type = bool, default=False)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    #Import network
    net = nt.GrainNet(block_nb = 2)
    net.load_state_dict(torch.load(args.model, map_location=torch.device(device)))
    net.eval()
    
    torch.manual_seed(args.seed)
    r = torch.tensor([args.grain_size])

    if not args.color:
        img = torchvision.io.read_image(args.input)/255.
        if img.shape[0]>1:
            img = torch.mean(img,axis=0)[None,:]

        img_out = net(torch.unsqueeze(img,0), r).detach().cpu()
    else:    
        img_r = torchvision.io.read_image(args.input)[0:1,:,:]/255.
        img_g = torchvision.io.read_image(args.input)[1:2,:,:]/255.
        img_b = torchvision.io.read_image(args.input)[2:3,:,:]/255.

        img_outr = net(torch.unsqueeze(img_r,0), r).detach().cpu()
        img_outg = net(torch.unsqueeze(img_g,0), r).detach().cpu()
        img_outb = net(torch.unsqueeze(img_b,0), r).detach().cpu()

        img_out = torch.cat((img_outr,img_outg,img_outb),axis=0)

    torchvision.utils.save_image(img_out,args.output)