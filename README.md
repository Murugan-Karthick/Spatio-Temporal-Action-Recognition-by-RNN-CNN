# Spatio-Temporal-Action-Recognition with a CNN-RNN Architecture

**Training a action recognizer with transfer learning and a recurrent model on the UCF101 dataset.**

## Demo

![alt text](https://github.com/Murugan-Karthick/Spatio-Temporal-Action-Recognition-by-RNN-CNN/blob/main/animation.gif)

We will be using the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php)
to build our action recognizer. The dataset consists of videos categorized into different actions, like

1. cricket shot, 
2. punching, 
3. biking, etc. 

A video consists of an ordered sequence of frames. Each frame contains *spatial*
information, and the sequence of those frames contains *temporal* information. To model
both of these aspects, we use a hybrid architecture that consists of convolutions
(for spatial processing) as well as recurrent layers (for temporal processing).
Specifically, we'll use a Convolutional Neural Network (CNN) and a Recurrent Neural
Network (RNN) consisting of [GRU layers].
