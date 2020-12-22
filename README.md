# Introduction to PyTorch
This repository contains notebooks for hands-on purpose during training session. All notebooks had been tested using CPU.

## Contents

#### [Chapter 1: Tensor](https://github.com/CertifaiAI/Intro_to_Pytorch/blob/main/solution/Chapter%201%20Tensors.ipynb)
- 1.1 Introduction to Tensors
- 1.2 Mathematical Operations on Tensors
- 1.3 Tensor Indexing, Slicing, Joining, Mutating
- 1.4 Tensor Objects Methods
- 1.5 Tensors on CPU and GPU

#### [Chapter 2: Autograd](https://github.com/CertifaiAI/Intro_to_Pytorch/blob/main/solution/Chapter%202%20Autograd.ipynb) 

- 2.1 Introduction to Autograd
- 2.2 Linear Regression Example

#### [Chapter 3: Build Your Neural Network with PyTorch](https://github.com/CertifaiAI/Intro_to_Pytorch/blob/main/solution/Chapter%203%20Neural%20Network.ipynb)
- 3.1 Dataloader
- 3.2 Build your First Neural Network (Subclassing nn.Module)
- 3.3 Build Your First Neural Network (Sequential Model)

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













