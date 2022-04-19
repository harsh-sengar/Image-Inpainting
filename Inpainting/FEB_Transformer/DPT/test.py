import os
import glob
import torch
import cv2
import argparse
import numpy as np
import util.io

from torchvision.transforms import Compose

img_names = glob.glob(os.path.join("images_DPT3/", "*"))
num_images = len(img_names)
for ind, img_name in enumerate(sorted(img_names)):
    img = util.io.read_image(img_name)
    img = np.array(img)
    print("img", img_name, np.min(img), np.max(img))