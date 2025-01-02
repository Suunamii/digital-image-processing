#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 18:06:02 2024

@author: gs
"""

import cv2 as cv
import matplotlib.pyplot as plt

imgH = cv.imread('hand.tif', cv.IMREAD_GRAYSCALE)
imgL = cv.imread('lena.tif', cv.IMREAD_GRAYSCALE)

handF = cv.medianBlur(imgH, 3)
lenaF = cv.medianBlur(imgL, 3)

plt.figure(figsize=(10, 8))


plt.subplot(221)
plt.title("Original - hand.tif")
plt.imshow(imgH, cmap='gray')
plt.axis("off")

plt.subplot(222)
plt.title("Gefiltert (Median) - hand.tif")
plt.imshow(handF, cmap='gray')
plt.axis("off")


plt.subplot(223)
plt.title("Original - lena.tif")
plt.imshow(imgL, cmap='gray')
plt.axis("off")

plt.subplot(224)
plt.title("Gefiltert (Median) - lena.tif")
plt.imshow(lenaF, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()