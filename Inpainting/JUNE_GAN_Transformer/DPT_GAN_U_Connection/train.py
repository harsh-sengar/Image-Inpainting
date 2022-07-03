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
import torch

from dataset import *
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch

import glob
import cv2

from torchvision.transforms import Compose
from torchinfo import summary
from dpt.models import DPTInpainting, ContextDiscriminator
from dpt.transforms import NormalizeImage

#from util.misc import visualize_attention
os.makedirs("saved_model", exist_ok=True)
os.makedirs("images_DPT_GAN_U_128", exist_ok=True)
torch.manual_seed(0)
cuda = True if torch.cuda.is_available() else False
os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_path", default="input", help="folder with input images")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="paris_train_original/input", help="name of the dataset")
parser.add_argument("-m", "--model_weights", default="saved_model/DPT_epoch_LR_1_100.pt", help="path to model weights")
parser.add_argument("-t", "--model_type", default="dpt_hybrid", help="model type [dpt_large|dpt_hybrid]",)
parser.add_argument("--absolute_depth", dest="absolute_depth", action="store_true")
parser.add_argument("--optimize", dest="optimize", action="store_true")
parser.add_argument("--no-optimize", dest="optimize", action="store_false")
parser.add_argument('--arc', type=str, choices=['celeba', 'places2', 'paris'], default='celeba')
parser.add_argument('--input_size', type=int, default=128)
parser.add_argument('--ld_input_size', type=int, default=64)
parser.add_argument('--max_holes', type=int, default=1)
parser.add_argument('--hole_min_w', type=int, default=24)
parser.add_argument('--hole_max_w', type=int, default=48)
parser.add_argument('--hole_min_h', type=int, default=24)
parser.add_argument('--hole_max_h', type=int, default=48)
parser.add_argument('--alpha', type=float, default=4e-3)
parser.add_argument('--steps_1', type=int, default=30000)
parser.add_argument('--steps_2', type=int, default=1000)
parser.add_argument('--steps_3', type=int, default=40000)
parser.add_argument('--bdivs', type=int, default=1)
parser.set_defaults(optimize=True)

args = parser.parse_args()

writer = SummaryWriter('runs/DPT_GAN_experiment')

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
# Tensor = torch.cuda.HalfTensor if cuda else torch.FloatTensor
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
n_epochs = 100
input_dim = 3
real_dim = 3
display_step = 50
batch_size = 8
lr = 1e-2

def train(save_model=True):

    print("initialize")
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    net_w = net_h = 128
    generator = DPTInpainting(
        path=None,
        backbone="vitb16_384",
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

    generator.to(device)
    discriminator.to(device)
    
    gen_opt =  torch.optim.SGD((generator.parameters()), lr = lr)
    dis_opt = torch.optim.SGD((discriminator.parameters()), lr = lr/50)

    mean_generator_loss = 0
    mean_discriminator_loss = 0
    G_losses = []
    D_losses = []
    cur_step = 0
    start_time = time.time()
    model_stats = summary(generator,input_size =(4,3,384,384), verbose=2)
    # print(model_stats)
    # exit(0)

    scheduler_g = torch.optim.lr_scheduler.MultiStepLR(gen_opt, milestones=[20000, 750000, 150000], gamma=0.1)

    for epoch in range(n_epochs):
        # Dataloader returns the batches
        generator.train()
        discriminator.train()

        gen_opt.zero_grad()
        dis_opt.zero_grad()

        # ---------------------
        #  Training Phase 1
        # ---------------------
        # Training Completion Network (Transformer)

        cur_step = 0
        pbar = tqdm(total=args.steps_1)
        while pbar.n < args.steps_1:
            for i, (imgs, _, _) in enumerate(train_dataloader):  
                gen_opt.zero_grad()
                
                # Configure input
                imgs = Variable(imgs.type(Tensor))
                
                # forward
                gen_output = generator(imgs)
                
                g_loss = mse_loss(gen_output, imgs)

                # backward
                g_loss.backward()
               
                # optimize
                gen_opt.step()
                scheduler_g.step()

                pbar.update()

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
                        save_image(imgs[i].data, "images_DPT_GAN_U_128/p1_%d_%d_%d.png" % (i, cur_step, epoch), nrow=1, normalize=True)
                        save_image((gen_output[i].data+1)/2, "images_DPT_GAN_U_128/p1_%d_%d_%d_gen_output.png" % (i, cur_step, epoch), nrow=1, normalize=False)
                    mean_generator_loss = 0
                cur_step += 1

                if pbar.n >= args.steps_1:
                    break
        pbar.close()

        gen_opt.zero_grad()
        dis_opt.zero_grad()

        # ================================================
        # Training Phase 3
        # ================================================
        # Train context discriminator

        cur_step = 0
        pbar = tqdm(total=args.steps_2)
        while pbar.n < args.steps_2:
            for i, (imgs, _, _) in enumerate(train_dataloader):  
                
                dis_opt.zero_grad()
                # fake forward

                hole_area_fake = gen_hole_area(
                    (args.ld_input_size, args.ld_input_size),
                    (imgs.shape[3], imgs.shape[2]))
                mask = gen_input_mask(
                    shape=(imgs.shape[0], 1, imgs.shape[2], imgs.shape[3]),
                    hole_size=(
                        (args.hole_min_w, args.hole_max_w),
                        (args.hole_min_h, args.hole_max_h)),
                    hole_area=hole_area_fake,
                    max_holes=args.max_holes).to(device)
                
                
                fake = torch.zeros((len(imgs), 1)).to(device)
                fake = Variable(fake.type(Tensor))
                imgs = Variable(imgs.type(Tensor))
                x_mask = imgs - imgs * mask
                x_mask = Variable(x_mask.type(Tensor))
                gen_output = generator(imgs)
                input_gd_fake = gen_output.detach().contiguous()
                input_ld_fake = crop(input_gd_fake, hole_area_fake)
                print(input_ld_fake.shape, input_gd_fake.shape, hole_area_fake)
                output_fake = discriminator((
                    input_ld_fake.to(device),
                    input_gd_fake.to(device)))
                loss_fake = adversarial_loss(output_fake, fake)

                # real forward
                hole_area_real = gen_hole_area(
                    (args.ld_input_size, args.ld_input_size),
                    (imgs.shape[3], imgs.shape[2]))
                real = torch.ones((len(imgs), 1)).to(device)
                real = Variable(real.type(Tensor))
                input_gd_real = imgs
                input_ld_real = crop(input_gd_real, hole_area_real)
                output_real = discriminator((input_ld_real, input_gd_real))
                loss_real = adversarial_loss(output_real, real)

                # reduce
                loss = (loss_fake + loss_real) / 2.

                # backward
                loss.backward()
                
                # optimize
                dis_opt.step()
                pbar.set_description('phase 2 | train loss: %.5f' % loss.cpu())
                pbar.update()

                D_losses.append(loss.item())
                mean_discriminator_loss += loss.item() 
                writer.add_scalar('discriminator training loss p2',
                            loss.item(),
                            epoch * len(train_dataloader) + i)

                if pbar.n >= args.steps_2:
                    break
        
        pbar.close()

        # ================================================
        # Training Phase 4
        # ================================================
        # Trained completion network and content discriminators 

        cur_step = 0
        mean_generator_loss = 0
        mean_discriminator_loss = 0
        G_losses = []
        D_losses = []
        pbar = tqdm(total=args.steps_3)
        while pbar.n < args.steps_3:
             for i, (imgs, _, _) in enumerate(train_dataloader): 
                gen_opt.zero_grad()
                dis_opt.zero_grad()

                # forward model_cd
                hole_area_fake = gen_hole_area(
                    (args.ld_input_size, args.ld_input_size),
                    (imgs.shape[3], imgs.shape[2]))
                mask = gen_input_mask(
                    shape=(imgs.shape[0], 1, imgs.shape[2], imgs.shape[3]),
                    hole_size=(
                        (args.hole_min_w, args.hole_max_w),
                        (args.hole_min_h, args.hole_max_h)),
                    hole_area=hole_area_fake,
                    max_holes=args.max_holes).to(device)

                # fake forward
                fake = torch.zeros((len(imgs), 1)).to(device)
                fake = Variable(fake.type(Tensor))
                imgs = Variable(imgs.type(Tensor))
                x_mask = imgs - imgs * mask
                x_mask = Variable(x_mask.type(Tensor))
                gen_output = generator(imgs)
                input_gd_fake = gen_output.detach().contiguous()
                input_ld_fake = crop(input_gd_fake, hole_area_fake)
                output_fake = discriminator((input_ld_fake, input_gd_fake))
                loss_cd_fake = adversarial_loss(output_fake, fake)

                # real forward
                hole_area_real = gen_hole_area(
                    (args.ld_input_size, args.ld_input_size),
                    (imgs.shape[3], imgs.shape[2]))
                real = torch.ones((len(imgs), 1)).to(device)
                real = Variable(real.type(Tensor))
                input_gd_real = imgs
                input_ld_real = crop(input_gd_real, hole_area_real)
                output_real = discriminator((input_ld_real, input_gd_real))
                loss_cd_real = adversarial_loss(output_real, real)

                # reduce
                d_loss = (loss_cd_fake + loss_cd_real) * alpha / 2.
                # print("output_fake", output_fake,"output_real", output_real,"loss_cd_fake", loss_cd_fake,"loss_cd_real", loss_cd_real)

                # backward Context discriminator
                d_loss.backward()
                
                # optimize
                dis_opt.step()

                # forward Completion model
                loss_cn_1 = mse_loss(gen_output, imgs )

                input_gd_fake = gen_output
                input_ld_fake = crop(input_gd_fake, hole_area_fake)
                output_fake = discriminator((input_ld_fake, (input_gd_fake)))
                loss_cn_2 = adversarial_loss(output_fake, real)

                # reduce
                g_loss = (loss_cn_1 + alpha * loss_cn_2) / 2.

                # backward Completion model
                g_loss.backward()
            
                # optimize
                gen_opt.step()
                pbar.set_description(
                    'phase 3 | train loss (cd): %.5f (cn): %.5f' % (
                        d_loss.cpu(),
                        g_loss.cpu()))
                pbar.update()
                
                # Save Losses for plotting later
                G_losses.append(g_loss.item())
                mean_generator_loss += g_loss.item() 

                D_losses.append(d_loss.item())
                mean_discriminator_loss += d_loss.item() 

                writer.add_scalar('discriminator training loss p4',
                            d_loss.item(),
                            epoch * len(train_dataloader) + i)

                writer.add_scalar('generator training loss p4',
                            g_loss.item(),
                            epoch * len(train_dataloader) + i)

                if cur_step % display_step == 0:
                    if cur_step > 0:
                        print(f"Epoch {epoch}: Step {cur_step}: Generator DPT loss: {mean_generator_loss/ display_step}, Discriminator Loss: {mean_discriminator_loss/ display_step}, Generator LR: {gen_opt.param_groups[0]['lr']}, Discriminator LR: {dis_opt.param_groups[0]['lr']}")
                    else:
                        print("Pretrained initial state")
                    # sample = torch.cat((imgs.data, masked_imgs.data, gen_parts.data), -2)
                    for i in range(len(imgs)):
                        save_image(imgs[i].data, "images_DPT_GAN_U_128/p4_%d_%d_%d_.png"  % (i, cur_step, epoch), nrow=1, normalize=True)
                        save_image((gen_output[i].data+1)/2, "images_DPT_GAN_U_128/p4_%d_%d_%d_gen_output.png"  % (i, cur_step, epoch), nrow=1, normalize=False)
                    mean_generator_loss = 0
                    mean_discriminator_loss = 0
                cur_step += 1

                if pbar.n >= args.steps_3:
                    break
        pbar.close()
        
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
        plt.savefig('plot/plot_DPT_GAN_%d_epochs_%d.png'%(batch_size, epoch)) 

        if save_model:
            gen_name = os.path.join("saved_model/", 'DPT_GAN_128_Gen_u_epoch_%d_.pt'%(epoch))
            dis_name = os.path.join("saved_model/", 'DPT_GAN_128_disc_u_epoch_%d_.pt'%(epoch))
            opt_name = os.path.join("saved_model/", 'optimizer_DPT_GAN__U_128_%d_.pt'%(epoch))

            torch.save(generator.state_dict(), gen_name)
            torch.save(discriminator.state_dict(), dis_name)
            torch.save({'gen': gen_opt.state_dict(),
                            'dis': dis_opt.state_dict()}, opt_name)

train()





