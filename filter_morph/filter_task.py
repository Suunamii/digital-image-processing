#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:25:50 2024

@author: gsu
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

origi = 'coins.png'
img = cv.imread(origi)
if img is None:
    raise FileNotFoundError("where is ur image?!!")
    
# -- Mittelwert Filter
mittelF = cv.blur(img, (5, 5))

# -- Gauss Filter
gaussF = cv.GaussianBlur(img, (5, 5), 0)

# -- Median Filter
medianF = cv.medianBlur(img, 5)

# -- Sobel Filter 
sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
sobelF = cv.magnitude(sobel_x, sobel_y)

# -- Laplace Filter
laplaceF = cv.Laplacian(img, cv.CV_64F)


plt.figure(figsize=(12, 8))

plt.subplot(231)

plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(232)
plt.title("Mittelwert Filter")
plt.imshow(mittelF, cmap='gray')
plt.axis('off')

plt.subplot(233)
plt.title("Gauss Filter")
plt.imshow(gaussF, cmap='gray')
plt.axis('off')

plt.subplot(234)
plt.title("Median Filter")
plt.imshow(medianF, cmap='gray')
plt.axis('off')

plt.subplot(235)
plt.title("Sobel Filter")
plt.imshow(sobelF, cmap='gray')
plt.axis('off')

plt.subplot(236)
plt.title("Laplace Filter")
plt.imshow(np.abs(laplaceF), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

"""
-- Mittelwert: einfach, reduziert noise , vanish contour
-- Gauss: much smoother than Mittelwert,  contouring not vanished
-- Median: more complex to calc., vsnish noise smooth
-- Sobel: greate for contouring, but more noise problem
-- Laplace: the most noise problem,  
"""
