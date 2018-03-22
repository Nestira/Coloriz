# Project Proposal: Automatic Colorization

Jiayue Zhang 1003555146 jyzhang.melanie@gmail.com
Yong Jiang 1003581402 yong.jiang@mail.utoronto.ca

Inspired by [Automatic Colorization](http://tinyclouds.org/colorize/) using CNN and one of it's application [Autocolored Animation](https://www.youtube.com/watch?v=V8AjYjXxno0) , We want to build a neural network that can differentiate objects in greyscale picture and colorize those objects automatically. We will use a predefined image classification model to extract features to color, and reconstruct it into RGB image directly. 

One of the limit mentioned in that project is that the size of the picture is fixed. If we have time, we want to break this limitation by resizing the picture before feed it into the network and then rescaling it after colorization.

## Papers to read: 
- [Visualizing and Understanding Convolutional Networks](http://arxiv.org/abs/1311.2901)
- [Hypercolumns for object segmentation and fine grained localization](http://arxiv.org/abs/1411.5752)
- [Multiple Object Recognition with Visual Attention](https://arxiv.org/abs/1412.7755)

## Software to use: Tensorflow


