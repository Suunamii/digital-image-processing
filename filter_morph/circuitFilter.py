#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 18:18:37 2024

@author: gs
"""


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

origi = "circuit.png"
img = cv.imread(origi, cv.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Where is your image?!")

laplaceF = np.array ([
    
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
    
    ])

highpass = cv.filter2D(img, -1, laplaceF)

img_sharp = cv.addWeighted(img, 1.0, highpass, 1.0, 0)


plt.figure(figsize=(12, 6))

plt.subplot(131)
plt.title("Originalbild")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(132)
plt.title("Hochpass-Filter Ergebnis")
plt.imshow(highpass, cmap='gray')
plt.axis("off")

plt.subplot(133)
plt.title("Geschärftes Bild")
plt.imshow(img_sharp, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()