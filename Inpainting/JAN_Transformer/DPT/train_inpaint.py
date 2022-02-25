from cmath import isnan
import torch
import argparse
import torchvision.datasets as dset
import numpy as np
import torchvision
import os
import time
import argparse
import os
import numpy as np
import math
import pandas as pd
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets as dset
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary

from dataset import *
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch

import glob
import cv2

from torchvision.transforms import Compose

from dpt.models import DPTDepthModel, DPTInpainting
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

#from util.misc import visualize_attention
os.makedirs("saved_model", exist_ok=True)
os.makedirs("images_DPT2", exist_ok=True)
torch.manual_seed(0)
cuda = True if torch.cuda.is_available() else False
os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_path", default="input", help="folder with input images")
parser.add_argument("-o", "--output_path", default="output_monodepth", help="folder for output images",)
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="paris_train_original/input", help="name of the dataset")
parser.add_argument("-m", "--model_weights", default="weights/dpt_hybrid.pt", help="path to model weights")
parser.add_argument("-t", "--model_type", default="dpt_hybrid", help="model type [dpt_large|dpt_hybrid]",)
parser.add_argument("--absolute_depth", dest="absolute_depth", action="store_true")
parser.add_argument("--optimize", dest="optimize", action="store_true")
parser.add_argument("--no-optimize", dest="optimize", action="store_false")
parser.set_defaults(optimize=True)
parser.set_defaults(kitti_crop=False)
parser.set_defaults(absolute_depth=False)

args = parser.parse_args()

default_models = {
        "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
}

if args.model_weights is None:
    args.model_weights = default_models[args.model_type]
args = parser.parse_args()

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr_decay = 5e-5
    lr = 0.0002
    lr = lr / (1.0 + lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# New parameters
recon_criterion = nn.L1Loss()
Tensor = torch.cuda.HalfTensor if cuda else torch.FloatTensor
n_epochs = 100
input_dim = 3
real_dim = 3
display_step = 50
batch_size = 8
lr = 1e-2
target_shape = 384
device = 'cuda'

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

def train(save_model=True):

    print("initialize")
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    if args.model_type == "dpt_large":  # DPT-Large
        net_w = net_h = 384
        model = DPTInpainting(
            path=args.model_weights,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif args.model_type == "dpt_hybrid":  # DPT-Hybrid
        net_w = net_h = 384
        model = DPTInpainting(
            path=None,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    transforms_ = [
    transforms.Resize((net_w, net_h), cv2.INTER_CUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
    transform = [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    
    dataloader = DataLoader(
        ImageDataset("%s" % args.dataset_name, transforms_=transforms_ , mask_size=64),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    ) 

    pretrained = True
    if pretrained:
        model = model.apply(weights_init)

    if device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)
    
    # for param in model.parameters():
    #     param.requires_grad = False

    # model.scratch.output_conv[0].weight.requires_grad = True
    # model.scratch.output_conv[2].weight.requires_grad = True
    # model.scratch.output_conv[4].weight.requires_grad = True

    gen_opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    mean_generator_loss = 0
    cur_step = 0
    start_time = time.time()
    G_losses = []
    scheduler_g = torch.optim.lr_scheduler.MultiStepLR(gen_opt, milestones=[2000, 10000, 25000, 100000], gamma=0.1)

    for epoch in range(n_epochs):
        # Dataloader returns the batches
        model.train()
        for i, (imgs, masked_imgs, masked_parts) in enumerate(dataloader):  

            # Configure input
            imgs = Variable(imgs.type(Tensor))
            masked_imgs = Variable(masked_imgs.type(Tensor))
            masked_parts = Variable(masked_parts.type(Tensor))

            if device == torch.device("cuda"):
                masked_imgs = masked_imgs.to(memory_format=torch.channels_last)
                masked_imgs = masked_imgs.half()

            if device == torch.device("cuda"):
                imgs = imgs.to(memory_format=torch.channels_last)
                imgs = imgs.half() 

            # print(imgs.shape)
            # -----------------
            #  Train Generator
            # -----------------
            gen_opt.zero_grad()

            # Generate a batch of images
            gen_parts = model((masked_imgs))
            # Adversarial and pixelwise loss

            # Total loss
            g_loss = recon_criterion(gen_parts, (imgs))
            # g_loss = torch.clamp(g_loss, min = -100, max =+100)

            if torch.isnan(g_loss):
                    print("min", torch.min(gen_parts).data, torch.min(masked_imgs), torch.min(imgs), torch.isnan(imgs).any())
                    print("max", torch.max(gen_parts).data, torch.max(masked_imgs), torch.max(imgs), torch.isnan(gen_parts).any())
                    exit(0)
             
            g_loss.backward()
            gen_opt.step()
            scheduler_g.step()
            
            # Save Losses for plotting later
            G_losses.append(g_loss.item())
            mean_generator_loss += g_loss.item() 
            ### Visualization code ###
            if cur_step % display_step == 0:
                if cur_step > 0:
                    print(f"Epoch {epoch}: Step {cur_step}: Generator DPT loss: {mean_generator_loss/ display_step}, Generator LR: {gen_opt.param_groups[0]['lr']}")
                else:
                    print("Pretrained initial state")
                sample = torch.cat((imgs.data, masked_imgs.data, gen_parts.data), -2)
                save_image(imgs.data, "images_DPT2/%d.png" % cur_step, nrow=1, normalize=True)
                save_image(masked_imgs.data, "images_DPT2/%d_mi.png" % cur_step, nrow=1, normalize=True)
                save_image(gen_parts.data, "images_DPT2/%d_gp.png" % cur_step, nrow=1, normalize=True)
                mean_generator_loss = 0
                print("min", torch.min(gen_parts), torch.min(masked_imgs), torch.min(imgs), torch.isnan(imgs).any())
                print("max", torch.max(gen_parts), torch.max(masked_imgs), torch.max(imgs), torch.isnan(gen_parts).any())
            cur_step += 1
        
    end_time = time.time()
    total_time = end_time - start_time
    print("Time taken : ", total_time)
    # You can change save_model to True if you'd like to save the model
    plt.figure(figsize=(10,5))
    plt.title("DPT pixel Loss During Training")
    plt.plot(G_losses,label="Generator Loss")
    plt.xlabel("iteration")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig('plot/plot_DPT_batch_%d_epoch_LR_2_%d.png'%(batch_size, n_epochs)) 
    if save_model:
        gen_name = os.path.join("saved_model/", 'DPT_epoch_LR_2_%d.pt'%n_epochs)
        opt_name = os.path.join("saved_model/", 'optimizer_DPT_epoch_LR_2_%d.pt'%n_epochs)

        torch.save(model.state_dict(), gen_name)
        torch.save(gen_opt.state_dict(), opt_name)

train()





