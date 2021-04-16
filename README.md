# Neural-Networks
*Authors: Mauro Ficorella, Valentina Sisti, Martina Turbessi*

Project for the course of Neural Networks from the master degree in "Engineering in Computer Science" at Sapienza

## Description

Tensorflow implementation of the paper "Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis", the paper can be found [**here**](https://arxiv.org/pdf/2101.04775v1.pdf).

## Prerequisites
* Windows/MacOS operating system;
* Python 3.8.x;
* Tensorflow 2.x;  
* In order to take the full advantage of CUDA computational power, you need a CUDA compatible hardware, CUDA toolkit 11 and cuDNN 8;
* You can recreate our environment through the provided environment.yml file.

## Usage
Example of usage in order to start the train:

`python Code/run.py --resolution 512 --batch 8 --diff-augment --FID`



