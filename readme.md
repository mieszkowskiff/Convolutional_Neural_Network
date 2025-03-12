# Convolutional Neural Network

This project is a part of course in DeepLearning at Warsaw University of Technology, summer semester 2024/2025.

## Dataset

The CINIC10 dataset (that should be stored under ```./data``` folder) can be downloaded from [here](https://www.kaggle.com/datasets/mengcius/cinic10).
The dataset contains $10$ classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) and is already divided into train, test and valid datasets. Each datapoint is a picture in resolution $32 \times 32$ with $3$ colour channels.

## Setup

```{Bash}
pip install numpy, matplotlib, torch
```