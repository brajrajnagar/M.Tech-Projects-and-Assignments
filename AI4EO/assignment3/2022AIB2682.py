# -*- coding: utf-8 -*-
"""
AIL861: Assignment 3
"""


##Import necessary libraries
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import torchvision.models as models

transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()]) 


## Define your function to find the similarity here
def findSimilarity(img1, img2):  ##Pass your arguments
    fm1 = torch.flatten(model(img1)).unsqueeze(0)
    cos_sim = []
    fm2 = model(img2)
    cols = fm2.size()[-1]
    rows = fm2.size()[-2]
    for i in range(cols):
        shifted_tensor = torch.roll(fm2, shifts=i, dims=-1)
        for j in range(rows):
            shifted_tensor = torch.roll(shifted_tensor, shifts=j, dims=-2)
            cos_sim.append(torch.nn.functional.cosine_similarity(fm1, torch.flatten(shifted_tensor).unsqueeze(0)).item())

    return max(cos_sim)


image1Path = transform(Image.open('1.png').convert('RGB').filter(ImageFilter.MedianFilter(size=3)))
image2Path = transform(Image.open('2.png').convert('RGB').filter(ImageFilter.MedianFilter(size=3)))
image3Path = transform(Image.open('3.png').convert('RGB').filter(ImageFilter.MedianFilter(size=3)))
image4Path = transform(Image.open('4.png').convert('RGB').filter(ImageFilter.MedianFilter(size=3)))
image5Path = transform(Image.open('5.png').convert('RGB').filter(ImageFilter.MedianFilter(size=3)))
image6Path = transform(Image.open('6.png').convert('RGB').filter(ImageFilter.MedianFilter(size=3)))

vgg16 = models.vgg16(weights = 'DEFAULT')
features = list(vgg16.features)
model = torch.nn.Sequential(*features)

##Read images and perform preprocessing if any


similarityScore = findSimilarity(image2Path, image5Path)    ## pass arguments for images 1 and 2
print('Similarity score between images 1 and 2')
print(similarityScore)

# similarityScore = findSimilarity()    ## pass arguments for images 2 and 3
# print('Similarity score between images 2 and 3')
# print(similarityScore)

# similarityScore = findSimilarity()    ## pass arguments for images 1 and 3
# print('Similarity score between images 1 and 3')
# print(similarityScore)

# similarityScore = findSimilarity()    ## pass arguments for images 1 and 4
# print('Similarity score between images 1 and 4')
# print(similarityScore)

# similarityScore = findSimilarity()    ## pass arguments for images 1 and 5
# print('Similarity score between images 1 and 5')
# print(similarityScore)

# similarityScore = findSimilarity()    ## pass arguments for images 1 and 6
# print('Similarity score between images 1 and 6')
# print(similarityScore)









