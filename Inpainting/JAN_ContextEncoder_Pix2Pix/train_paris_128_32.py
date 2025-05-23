
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

from dataset import *
from models import *
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images_contextencoder_paris", exist_ok=True)
os.makedirs("saved_model", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="paris_train_original/input", help="name of the dataset")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--mask_size", type=int, default=32, help="size of random mask")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=50, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
patch_h, patch_w = int(opt.mask_size / 2 ** 3), int(opt.mask_size / 2 ** 3)
patch = (1, patch_h, patch_w)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Loss function
adversarial_loss = torch.nn.MSELoss()
pixelwise_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
generator = Generator(channels=opt.channels)
discriminator = Discriminator(channels=opt.channels)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Dataset loader
transforms_ = [
    transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset("%s" % opt.dataset_name, transforms_=transforms_, mask_size=32),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
test_dataloader = DataLoader(
    ImageDataset("%s" % opt.dataset_name, transforms_=transforms_, mask_size=32, mode="val"),
    batch_size=12,
    shuffle=True,
    num_workers=1,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def save_sample(batches_done):
    samples, masked_samples, i = next(iter(test_dataloader))
    samples = Variable(samples.type(Tensor))
    masked_samples = Variable(masked_samples.type(Tensor))
    i = i[0].item()  # Upper-left coordinate of mask
    # Generate inpainted image
    gen_mask = generator(masked_samples)
    filled_samples = masked_samples.clone()
    filled_samples[:, :, i : i + opt.mask_size, i : i + opt.mask_size] = gen_mask
    # Save sample
    sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
    save_image(sample, "images_contextencoder_paris/%d.png" % batches_done, nrow=6, normalize=True)


# ----------
#  Training
# ----------
G_losses = []
D_losses = []
L_losses = []

losses = pd.DataFrame(columns=['G', 'D', 'L'])

for epoch in range(opt.n_epochs):
    for i, (imgs, masked_imgs, masked_parts) in enumerate(dataloader):
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)
       
        # Configure input
        imgs = Variable(imgs.type(Tensor))
        masked_imgs = Variable(masked_imgs.type(Tensor))
        masked_parts = Variable(masked_parts.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_parts = generator(masked_imgs)
        # Adversarial and pixelwise loss
        g_adv = adversarial_loss(discriminator(gen_parts), valid)
        g_pixel = pixelwise_loss(gen_parts, masked_parts)
        # Total loss
        g_loss = 0.001 * g_adv + 0.999 * g_pixel

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(masked_parts), valid)
        fake_loss = adversarial_loss(discriminator(gen_parts.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        errG = g_adv.item()
        errD = d_loss.item()
        errL = g_pixel.item()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_adv.item(), g_pixel.item())
        )

        # Generate sample at sample interval
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_sample(batches_done)
             
        # Save Losses for plotting later
        G_losses.append(errG)
        D_losses.append(errD)
        L_losses.append(errL)

    losses_iter = pd.DataFrame({"G":G_losses, "D": D_losses, "L":L_losses})
    losses.append(losses_iter)

    # ----------
    #  Save generators, discriminators and optimizers
    # ----------
    gen_name = os.path.join("saved_model/", 'gen_paris_128_32_batch_16_epoch_200.pt')
    dis_name = os.path.join("saved_model/", 'dis_paris_128_32_batch_16_epoch_200.pt')
    opt_name = os.path.join("saved_model/", 'optimizer_paris_128_32_batch_16_epoch_200.pt')

    torch.save(generator.state_dict(), gen_name)
    torch.save(discriminator.state_dict(), dis_name)
    torch.save({'gen': optimizer_G.state_dict(),
                        'dis': optimizer_D.state_dict()}, opt_name)


    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.plot(L_losses,label="L")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    # plt.legend()
    # plt.show()
    plt.savefig('plot/plot_paris_128_32_epoch_batch_16_epoch%f.pdf'%epoch)  

gen_name = os.path.join("saved_model/", 'gen_paris_128_32.pt')
dis_name = os.path.join("saved_model/", 'dis_paris_128_32.pt')
opt_name = os.path.join("saved_model/", 'optimizer_paris_128_32.pt')

torch.save(generator.state_dict(), gen_name)
torch.save(discriminator.state_dict(), dis_name)
torch.save({'gen': optimizer_G.state_dict(),
                        'dis': optimizer_D.state_dict()}, opt_name)

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.plot(L_losses,label="L")
plt.xlabel("epochs")
plt.ylabel("Loss")
# plt.legend()
# plt.show()
plt.savefig('plot/plot_paris_128_32_batch_16_epoch_200.pdf') 
losses.to_csv('loss_paris_128_32_batch_16_epoch_200.csv') 