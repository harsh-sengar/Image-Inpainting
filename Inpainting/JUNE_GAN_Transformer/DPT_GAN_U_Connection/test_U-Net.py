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
import argparse
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import math
import pandas as pd
import torchvision.transforms as transforms

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

from dpt.models import DPTInpainting, ContextDiscriminator
from dpt.transforms import NormalizeImage

#from util.misc import visualize_attention
os.makedirs("saved_model", exist_ok=True)
os.makedirs("images_DPT_U_384", exist_ok=True)
os.makedirs("test_DPT_U_384", exist_ok=True)
os.makedirs("val_DPT_U_384", exist_ok=True)
torch.manual_seed(0)
cuda = True if torch.cuda.is_available() else False
os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_path", default="input", help="folder with input images")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="paris_train_original/input", help="name of the dataset")
parser.add_argument("--test_dataset_name", type=str, default="paris_train_original/paris_eval/paris_eval_gt", help="name of the dataset")
parser.add_argument("-m", "--model_weights", default="saved_model/DPT_gen_U_384_new_p4_28.pt", help="path to model weights")
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
parser.add_argument('--steps_1', type=int, default=10000)
parser.add_argument('--steps_2', type=int, default=10000)
parser.add_argument('--steps_3', type=int, default=1000)
parser.add_argument('--steps_4', type=int, default=40000)
parser.add_argument('--bdivs', type=int, default=1)
parser.set_defaults(optimize=True)

args = parser.parse_args()

writer = SummaryWriter('runs/DPT_U_experiment')

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
pixelwise_loss = nn.MSELoss()
adversarial_loss = BCELoss()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
n_epochs = 100
input_dim = 3
real_dim = 3
display_step = 50
batch_size = 8
lr = 1e-2
target_shape = 384

def save_sample(generator, val_dataloader, epoch, mode = "val"):
    for i, (imgs, _, _) in enumerate(val_dataloader):  
        hole_area = gen_hole_area(
            (args.ld_input_size, args.ld_input_size),
            (imgs.shape[3], imgs.shape[2]))
        mask = gen_input_mask(
            shape=(imgs.shape[0], 1, imgs.shape[2], imgs.shape[3]),
            hole_size=(
                (args.hole_min_w, args.hole_max_w),
                (args.hole_min_h, args.hole_max_h)),
            hole_area=hole_area,
            max_holes=args.max_holes).to(device)
        
        # Configure input
        imgs = Variable(imgs.type(Tensor))
        x_mask = imgs - imgs * mask
        x_mask = Variable(x_mask.type(Tensor))

        # forward
        gen_output = generator(x_mask)
        
        if(mode == "val"):
            save_image(imgs.data, "val_DPT_U_384/val_%d_%d.png" % (i, epoch), nrow=1, normalize=True)
            save_image((gen_output.data+1)/2, "val_DPT_U_384/val_%d_%d_gen_output.png" % (i, epoch), nrow=1, normalize=False)
            save_image(x_mask.data, "val_DPT_U_384/val_%d_%d_mask.png" % (i, epoch), nrow=1, normalize=True)
        if(mode == "test"):
            save_image(imgs.data, "test_DPT_U_384/test_%d_%d.png" % (i, epoch), nrow=1, normalize=True)
            save_image((gen_output.data+1)/2, "test_DPT_U_384/test_%d_%d_gen_output.png" % (i, epoch), nrow=1, normalize=False)
            save_image(x_mask.data, "test_DPT_U_384/test_%d_%d_mask.png" % (i, epoch), nrow=1, normalize=True)

def train(save_model=True):

    print("initialize")
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    net_w = net_h = 384 
    generator=DPTInpainting(
        path=args.model_weights,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )

    discriminator = ContextDiscriminator(local_input_shape=(3, args.ld_input_size, args.ld_input_size),
    global_input_shape=(3, args.input_size, args.input_size),
    arc=args.arc)
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    transforms_ = [
    transforms.Resize((net_w, net_h), cv2.INTER_CUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    
    train_dataloader = DataLoader(
        ImageDataset("%s" % args.dataset_name, transforms_=transforms_ , mask_size=64),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    ) 

    val_dataloader = DataLoader(
        ImageDataset("%s" % args.dataset_name, transforms_=transforms_ , mask_size=64, mode="val"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    ) 

    test_dataloader = DataLoader(
        ImageDataset("%s" % args.test_dataset_name, transforms_=transforms_ , mask_size=64, mode="test"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    ) 

    generator.to(device)
    discriminator.to(device)

    # model_stats = summary(generator.unet,input_size =(6,384,384))
    # print(model_stats)
    # exit(0)

    cur_step = 0
    start_time = time.time()

    # model_stats = summary(generator,input_size =(3,384,384))
    # print(model_stats)
    # exit(0)

    iteration = 0
    for epoch in range(n_epochs):
        # Dataloader returns the batches
        generator.eval()
        discriminator.eval()

        # ---------------------
        #  Training Phase 1
        # ---------------------
        # Training Completion Network (Transformer)

        cur_step = 0
        pbar = tqdm(total=args.steps_1)
        for i, (imgs, _, _) in enumerate(train_dataloader):  
            hole_area = gen_hole_area(
                (args.ld_input_size, args.ld_input_size),
                (imgs.shape[3], imgs.shape[2]))
            mask = gen_input_mask(
                shape=(imgs.shape[0], 1, imgs.shape[2], imgs.shape[3]),
                hole_size=(
                    (args.hole_min_w, args.hole_max_w),
                    (args.hole_min_h, args.hole_max_h)),
                hole_area=hole_area,
                max_holes=args.max_holes).to(device)
            
            # Configure input
            imgs = Variable(imgs.type(Tensor))
            x_mask = imgs - imgs * mask
            x_mask = Variable(x_mask.type(Tensor))
            # forward
            gen_output = generator(x_mask)

            end_time = time.time()
            total_time = end_time - start_time
            print("Time taken : ", total_time)

            if cur_step % display_step == 0:
                if cur_step > 0:
                    print(f"Epoch {epoch}: Step {cur_step}: Generator DPT loss Image: {mean_generator_loss/ display_step}")
                else:
                    print("Pretrained initial state")

                save_sample(generator, val_dataloader, epoch, mode="val")
                save_sample(generator, test_dataloader, epoch, mode = "test")
                save_image(imgs.data, "images_DPT_U_384/p1_%d_%d_%d.png" % (i, cur_step, epoch), nrow=1, normalize=True)
                save_image((gen_output.data+1)/2, "images_DPT_U_384/p1_%d_%d_%d_gen_output.png" % (i, cur_step, epoch), nrow=1, normalize=False)
                save_image(x_mask.data, "images_DPT_U_384/p1_%d_%d_%d_mask.png" % (i, cur_step, epoch), nrow=1, normalize=True)
                mean_generator_loss = 0
            cur_step += 1
            iteration += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        print("Time taken : ", total_time)

train()





