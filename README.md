# Deep Learning Nanodegree Foundation
[![watch_video](https://d2mxuefqeaa7sj.cloudfront.net/s_D16C37C27D7DCCF16E416DD259155C3A4612061618EE343DACE65282E93C5FF3_1551803331348_Screenshot+2019-03-05+at+17.28.16.png)](https://www.youtube.com/watch?v=JSulOaFMJPI)

This repository contains material related to Udacity's [Deep Learning Nanodegree Foundation program](https://www.udacity.com/course/deep-learning-nanodegree--nd101) that was implemented by me. 

It consists of a bunch of tutorial notebooks for various deep learning topics. In most cases, the notebooks leads through implementing models such as convolutional networks, recurrent networks, and GANs. There are other topics covered such as weight intialization and batch normalization. There are also notebooks used as projects for the Nanodegree program. 

**It also includes Siraj Raval's youtube [Deep Learning](https://www.youtube.com/watch?v=vOppzHpvTiQ&list=PL2-dafEMk2A7YdKv4XfKpfbTH5z6rEEj3) tutorials videos when he collaborated with Udacity in the first cohort of this course.**

## Table Of Contents

### Tutorials

* Sentiment Analysis with Numpy: Andrew Trask leads you through building a sentiment analysis model, predicting if some text is positive or negative.

* Intro to TensorFlow: Starting building neural networks with Tensorflow.

* Weight Intialization: Explore how initializing network weights affects performance.

* Autoencoders: Build models for image compression and denoising, using feed-forward and convolution networks in TensorFlow.

* Transfer Learning (ConvNet). In practice, most people don't train their own large networkd on huge datasets, but use pretrained networks such as VGGnet. Here you'll use VGGnet to classify images of flowers without training a network on the images themselves.

* Intro to Recurrent Networks (Character-wise RNN): Recurrent neural networks are able to use information about the sequence of data, such as the sequence of characters in text.

* Embeddings (Word2Vec): Implement the Word2Vec model to find semantic representations of words for use in natural language processing.

* Sentiment Analysis RNN: Implement a recurrent neural network that can predict if a text sample is positive or negative.

* Tensorboard: Use TensorBoard to visualize the network graph, as well as how parameters change through training.

* Reinforcement Learning (Q-Learning): Implement a deep Q-learning network to play a simple game from OpenAI Gym.

* Sequence to sequence: Implement a sequence-to-sequence recurrent network.

* Batch normalization: Learn how to improve training rates and network stability with batch normalizations.

* Generative Adversatial Network on MNIST: Train a simple generative adversarial network on the MNIST dataset.

* Deep Convolutional GAN (DCGAN): Implement a DCGAN to generate new images based on the Street View House Numbers (SVHN) dataset.

* Intro to TFLearn: A couple introductions to a high-level library for building neural networks.

### Projects

* Your First Neural Network: Implement a neural network in Numpy to predict bike rentals.

* Image classification: Build a convolutional neural network with TensorFlow to classify CIFAR-10 images.

* Text Generation: Train a recurrent neural network on scripts from The Simpson's (copyright Fox) to generate new scripts.

* Machine Translation: Train a sequence to sequence network for English to French translation (on a simple dataset)

* Face Generation: Use a DCGAN on the CelebA dataset to generate images of novel and realistic human faces.

### Dependencies

Each directory has a requirements.txt describing the minimal dependencies required to run the notebooks in that directory.

#### pip
To install these dependencies with pip, you can issue pip3 install -r requirements.txt.

#### Conda Environments
You can find Conda environment files for the Deep Learning program in the environments folder. Note that environment files are platform dependent. Versions with tensorflow-gpu are labeled in the filename with "GPU".
