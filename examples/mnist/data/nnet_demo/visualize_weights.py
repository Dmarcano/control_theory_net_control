######################################################################
# visualize_weights.py - Visualize weights for neural network demo
# Writen for ENGR 027 - Computer Vision
# Matt Zucker 2020-2021
######################################################################

import cv2
import numpy as np

with np.load('mnist_weights_quantized.npz') as npzfile:
    weights = npzfile[npzfile.files[0]]

assert weights.shape == (785, 300) and weights.dtype == np.int8

weights = weights[:-1, :].transpose()
weights = weights.reshape(300, 28, 28).astype(np.float32)

rows = 20
cols = 15

size = 28

margin = 2

height = (size + margin)*rows + margin
width = (size + margin)*cols + margin

output = np.zeros((height, width), dtype=np.uint8)

row = 0
col = 0

for wimage in weights:
    
    wmin = wimage.min()
    wmax = wimage.max()
    display = (wimage - wmin) / (wmax - wmin)
    display = (display*255).astype(np.uint8)

    y = row * (size+margin) + margin
    x = col * (size+margin) + margin

    output[y:y+size, x:x+size] = display

    col += 1
    if col >= cols:
        col = 0
        row += 1

cv2.imwrite('weights.png', output)    
