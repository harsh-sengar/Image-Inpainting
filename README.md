# Image-Inpainting

[Download / Image Inpainting](https://drive.google.com/file/d/1wp72AaUtkZKqihni05VOCZcXe_-h7zW6/view?usp=sharing)

# Description
Image Inpainting(i.e., image completion or image hole-filling), which is a fundamental computer vision problem which is used for image editing in real world. Numerous approaches are proposed and performed well on many datasets but there exists a serious limitation of solving this complex problem with high resolution images. We introduce Transformer based generative adversarial network, Architecture that leverages vision transformer along with convolutions in the generator backbone for generating high resolution images with large mask. The model is two stage architecture in which stage one is Dense Prediction Transformer based Generative Adversarial Network which generates the missing region and the second stage is refinement network using U-Net architecture. We used skip connections between both stages to ensure the communication at different feature map level.

# Required libraries
* Python3
* PyTorch
* torchvision
* Numpy
* opencv
<br/>
Make sure to install the above libraries and finally run the scripts
