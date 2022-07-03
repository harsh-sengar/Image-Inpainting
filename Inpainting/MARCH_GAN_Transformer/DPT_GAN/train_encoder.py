from cmath import isnan
from dis import dis
from operator import ge
import torch
import argparse
import torchvision.datasets as dset
from torch.nn.functional import mse_loss
from torch.nn import BCELoss, DataParallel
import numpy as np
from torch.optim import Adadelta, Adam
import torchvision
import os
import time
from tqdm import tqdm

import numpy as np
import math
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from PIL import Image
from utils import (
    gen_input_mask,
    gen_hole_area,
    crop,
    sample_random_batch,
    poisson_blend,
)

from torch.utils.data import DataLoader
from torchvision import datasets as dset
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from dataset import *
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch

import glob
import cv2

from torchvision.transforms import Compose

from dpt.models import Generator1, DPTInpainting, ContextDiscriminator, Generator
from dpt.transforms import NormalizeImage

#from util.misc import visualize_attention
os.makedirs("saved_model", exist_ok=True)
os.makedirs("images_encoder", exist_ok=True)
os.makedirs("images_encoder_test", exist_ok=True)
torch.manual_seed(0)
cuda = True if torch.cuda.is_available() else False
os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_path", default="input", help="folder with input images")
parser.add_argument("-o", "--output_path", default="output_monodepth", help="folder for output images",)
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="paris_train_original/input", help="name of the dataset")
parser.add_argument("-m", "--model_weights", default="saved_model/DPT_epoch_LR_1_100.pt", help="path to model weights")
parser.add_argument("-t", "--model_type", default="dpt_hybrid", help="model type [dpt_large|dpt_hybrid]",)
parser.add_argument("--absolute_depth", dest="absolute_depth", action="store_true")
parser.add_argument("--optimize", dest="optimize", action="store_true")
parser.add_argument("--no-optimize", dest="optimize", action="store_false")
parser.add_argument('--arc', type=str, choices=['celeba', 'places2', 'paris'], default='celeba')
parser.add_argument('--input_size', type=int, default=384)
parser.add_argument('--ld_input_size', type=int, default=96)
parser.add_argument('--max_holes', type=int, default=1)
parser.add_argument('--hole_min_w', type=int, default=48)
parser.add_argument('--hole_max_w', type=int, default=96)
parser.add_argument('--hole_min_h', type=int, default=48)
parser.add_argument('--hole_max_h', type=int, default=96)
parser.add_argument('--alpha', type=float, default=4e-3)
parser.add_argument('--steps_1', type=int, default=30000)
parser.add_argument('--steps_2', type=int, default=1000)
parser.add_argument('--steps_3', type=int, default=40000)
parser.add_argument('--bdivs', type=int, default=1)
parser.set_defaults(optimize=True)

args = parser.parse_args()

writer = SummaryWriter('runs/DPT_encoder_experiment')

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr_decay = 5e-5
    lr = 0.0002
    lr = lr / (1.0 + lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# New parameters
device = 'cuda'
alpha = torch.tensor(
        args.alpha,
        dtype=torch.float32).to(device)
pixelwise_loss = torch.nn.L1Loss()
adversarial_loss = BCELoss()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
n_epochs = 200
input_dim = 3
real_dim = 3
display_step = 50
batch_size = 8
lr = 1e-2
target_shape = 384

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

    net_w = net_h = 384
    generator = Generator1(channels = 3)
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    transforms_ = [
    transforms.Resize((net_w, net_h), cv2.INTER_CUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    
    train_dataloader = DataLoader(
        ImageDataset("%s" % args.dataset_name, transforms_=transforms_ , mask_size=16),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    ) 

    test_dataloader = DataLoader(
        ImageDataset("%s" % args.dataset_name, transforms_=transforms_, mask_size=16, mode="val"),
        batch_size=1,
        shuffle=True,
        num_workers=1,
    )

    # generator = generator.apply(weights_init)

    generator.to(device)

    gen_opt =  torch.optim.SGD((generator.parameters()), lr = lr)

    mean_generator_loss = 0
    G_losses = []
    cur_step = 0
    start_time = time.time()

    scheduler_g = torch.optim.lr_scheduler.MultiStepLR(gen_opt, milestones=[30000, 150000], gamma=0.1)

    for epoch in range(n_epochs):
        # Dataloader returns the batches
        generator.train()

        # ---------------------
        #  Training Phase 1
        # ---------------------
        # Training Completion Network (Transformer)

        for i, (imgs, _, _) in enumerate(train_dataloader):  
            gen_opt.zero_grad()
            # Configure input
            imgs = Variable(imgs.type(Tensor))
            # forward
            gen_output = generator(imgs)
            g_loss = pixelwise_loss(gen_output, imgs)
            g_loss.backward()
            
            # optimize
            gen_opt.step()
            scheduler_g.step()

            G_losses.append(g_loss.item())
            mean_generator_loss += g_loss.item() 

            writer.add_scalar('Generator training loss p1',
                        g_loss.item(),
                        epoch * len(train_dataloader) + i)

            if cur_step % display_step == 0:
                if cur_step > 0:
                    print(f"Epoch {epoch}: Step {cur_step}: Generator DPT loss Image: {mean_generator_loss/ display_step} : Generator LR: {gen_opt.param_groups[0]['lr']}")
                else:
                    print("Pretrained initial state")
                for i in range(len(imgs)):
                    save_image(imgs[i].data, "images_encoder/p1_%d_%d_%d.png" % (i, cur_step, epoch), nrow=1, normalize=True)
                    save_image((gen_output[i].data+1)/2, "images_encoder/p1_%d_%d_%d_gen_output.png" % (i, cur_step, epoch), nrow=1, normalize=False)
                mean_generator_loss = 0
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
        plt.savefig('plot/plot_encoder_%d_epochs_%d.png'%(batch_size, epoch)) 


        if save_model:
            gen_name = os.path.join("saved_model/", 'DPT_encoder_epoch_%d_.pt'%(epoch))
            dis_name = os.path.join("saved_model/", 'DPT_encoder_epoch_%d_.pt'%(epoch))
            opt_name = os.path.join("saved_model/", 'optimizer_encoder_%d_.pt'%(epoch))

            for i, (samples, _, _) in enumerate(test_dataloader):  
                samples = Variable(samples.type(Tensor))
                gen_mask = generator(samples)
                # Save sample
                save_image(samples.data, "images_encoder_test/sample_%d_%d.png" % (i, epoch), nrow=1, normalize=True)
                save_image((gen_mask+1)/2, "images_encoder_test/t_%d_%d.png" % (i, epoch), nrow=1, normalize=False)

            torch.save(generator.state_dict(), gen_name)
            torch.save({'gen': gen_opt.state_dict()}, opt_name)

train()





