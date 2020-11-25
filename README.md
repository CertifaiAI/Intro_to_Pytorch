# Introduction to PyTorch
<p>
  <p align="center">
    <a href="https://github.com/CertifaiAI/Intro_to_Pytorch/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/CertifaiAI/Intro_to_Pytorch.svg">
    </a>
    <a href="Discord">
        <img alt="Discord" src="https://img.shields.io/discord/699181979316387842?color=red">
    </a>
    <a href="https://certifai.ai">
        <img alt="Documentation" src="https://img.shields.io/website/http/certifai.ai.svg?color=ff69b4">
    </a>
    <a href="https://github.com/CertifaiAI/Intro_to_Pytorch/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/CertifaiAI/Intro_to_Pytorch.svg">
    </a>
</p>

A working repository on [Introduction to Pytorch](https://docs.google.com/document/d/1St7ZU7MzNR-It4zpJIqc6nSNtRiMYFb9ZKf5EaLwuYI/edit?usp=sharing) course.

<!-- 
[![](https://sourcerer.io/fame/chiaweilim/skymindglobal/Intro_to_Pytorch/images/0)](https://sourcerer.io/fame/chiaweilim/skymindglobal/Intro_to_Pytorch/links/0)[![](https://sourcerer.io/fame/chiaweilim/skymindglobal/Intro_to_Pytorch/images/1)](https://sourcerer.io/fame/chiaweilim/skymindglobal/Intro_to_Pytorch/links/1)[![](https://sourcerer.io/fame/chiaweilim/skymindglobal/Intro_to_Pytorch/images/2)](https://sourcerer.io/fame/chiaweilim/skymindglobal/Intro_to_Pytorch/links/2)[![](https://sourcerer.io/fame/chiaweilim/skymindglobal/Intro_to_Pytorch/images/3)](https://sourcerer.io/fame/chiaweilim/skymindglobal/Intro_to_Pytorch/links/3)[![](https://sourcerer.io/fame/chiaweilim/skymindglobal/Intro_to_Pytorch/images/4)](https://sourcerer.io/fame/chiaweilim/skymindglobal/Intro_to_Pytorch/links/4)[![](https://sourcerer.io/fame/chiaweilim/skymindglobal/Intro_to_Pytorch/images/5)](https://sourcerer.io/fame/chiaweilim/skymindglobal/Intro_to_Pytorch/links/5)[![](https://sourcerer.io/fame/chiaweilim/skymindglobal/Intro_to_Pytorch/images/6)](https://sourcerer.io/fame/chiaweilim/skymindglobal/Intro_to_Pytorch/links/6)[![](https://sourcerer.io/fame/chiaweilim/skymindglobal/Intro_to_Pytorch/images/7)](https://sourcerer.io/fame/chiaweilim/skymindglobal/Intro_to_Pytorch/links/7)
 -->
 
This repository contains notebooks for hands-on purpose during training session. All notebooks had been tested using CPU.

## Contents

#### [Chapter 1: Tensor](https://github.com/CertifaiAI/Intro_to_Pytorch/blob/main/solution/Chapter%201%20Tensors.ipynb)
- 1.1 Introduction to Tensors
- 1.2 Mathematical Operations on Tensors
- 1.3 Tensor Indexing, Slicing, Joining, Mutating
- 1.4 Tensor Objects Methods
- 1.5 Tensors on CPU and GPU

#### [Chapter 2: Autograd](https://github.com/CertifaiAI/Intro_to_Pytorch/blob/main/solution/Chapter_2.ipynb) 
- 2.1 Introduction to Autograd
- 2.2 Linear Regression Example

#### [Chapter 3: Build Your Neural Network with PyTorch](https://github.com/CertifaiAI/Intro_to_Pytorch/blob/main/solution/Chapter%203%20Neural%20Network.ipynb)
- 3.1 Dataloader
- 3.2 Build your First Neural Network (Subclassing nn.Module)
- 3.3 Build Your First Neural Network (Sequential Model)

#### [Chapter 4: Convolutional Neural Network]()
- 4.1 Image Transformation Pipeline
- 4.2 Build Your First Simple CNN Classifier (Pytorch Sequential Model)
- 4.3 Model Improvement by using Deeper Network
- 4.4 Building Your ResNet18 (Intro to Model Building with Class)
- 4.5 Transfer Learning

#### [Chapter 5: Recurrent Neural Network]()
- 5.1 Data preparation
- 5.2 Build an RNN model
- 5.3 Example
  
#### [Chapter 6: Advanced Computer Vision]()
- 6.1 
- 6.2 
- 6.3 

## Built with
- PyTorch 1.6.0
- CUDA 10.2

## Getting Started

### Install Anaconda Individual Edition

Download and install [Anaconda](https://www.anaconda.com/products/individual).

### Environment Setup

Setup the conda environment by

```
conda env create -f environment.yml
```

The environment setup will take some time to download required modules.

### GPU Setup (__*optional*__)
Follow the instructions below if you plan to use GPU setup.
1. Install CUDA and cuDNN
    Requirements:
   -  [CUDA 10.2](https://developer.nvidia.com/cuda-10.2-download-archive)
   -  [cuDNN 7.6](https://developer.nvidia.com/rdp/cudnn-archive)
   
> Step by step installation guides can be found [here](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn_765/cudnn-install/index.html#install-windows).

2. If you like to use different version of CUDA, please install appropriate **cudatoolkit** module by enter `conda install cudatoolkit=CUDA_VERSION`

```
conda install cudatoolkit=10.2
```

## Usage
All examples are separated into [training](https://github.com/CertifaiAI/Intro_to_Pytorch/tree/main/training) and [solution](https://github.com/CertifaiAI/Intro_to_Pytorch/tree/main/solution) folders.

All notebooks in **training** folder have few lines commented out so that they can be taught and demonstrated in the class. The **solution** folder contains the un-commented version for every line of codes.

## Known Issues
- 
-
-















