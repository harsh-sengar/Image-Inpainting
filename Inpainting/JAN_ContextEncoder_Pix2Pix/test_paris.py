
import argparse
import os
import glob
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets as dset
from torch.autograd import Variable
from matplotlib import cm

from dataset import *
from models import *

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="saved_model/gen_paris_128_32.pt", help="size of the batches")
parser.add_argument("--path", type=str, default="paris_train_original/input/", help="name of the dataset")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--mask_size", type=int, default=32, help="size of random mask")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
args = parser.parse_args()

os.makedirs("output", exist_ok=True)


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
transforms_ = [
    transforms.Resize((args.img_size, args.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
] 

def main():
    
    generator = Generator(channels=args.channels)
    generator.load_state_dict(torch.load(args.model_path))

    if cuda:
        generator.cuda()

    test_dataloader = DataLoader(
    ImageDataset("/data/hrshsengar/Inpainting/%s" % args.path, transforms_=transforms_, mask_size=32, mode="val"),
    batch_size=1,
    shuffle=False,
    num_workers=1,
    )

    j = 0
    for i, (imgs, masked_imgs, masked_parts) in enumerate(test_dataloader):
        # Configure input
        imgs = Variable(imgs.type(Tensor))
        masked_imgs = Variable(masked_imgs.type(Tensor))
        masked_parts = Variable(masked_parts.type(Tensor))
        m = int(masked_parts[0].item())
        # Generate a batch of images
        gen_parts = generator(masked_imgs)
        filled_samples = masked_imgs.clone()
        filled_samples[:, :, m : m + args.mask_size, m : m + args.mask_size] = gen_parts
        output = torch.cat((filled_samples, imgs ), 3)
        
        save_image(output, "paris_train_original/outputPix2Pix/pix2pix/%d.png"%j, normalize=True)
        j = j + 1

if __name__ == '__main__':
    main()