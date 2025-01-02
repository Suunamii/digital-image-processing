
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 18:41:29 2024

@author: gsu
"""
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.feature import blob_log
from math import sqrt


source = "coins.png"
img = cv.imread(source)
if img is None:
    raise FileNotFoundError("Where Ur Image??!")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(gray, (5, 5), 0)


_, binary = cv.threshold(blur, 90, 255, cv.THRESH_BINARY_INV)


binary_inverted = cv.bitwise_not(binary)  

# blob detection (Laplacian of Gaussian)
blobs_log = blob_log(binary_inverted, max_sigma=50, num_sigma=40, threshold=0.5)

# to adjust the radius for blobs
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)


num_coins = len(blobs_log)
print(f"Number of coins detected: {num_coins}")



fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].set_title("Original")
axes[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
axes[0].axis("off")


axes[1].set_title(f"Detected Coins: {num_coins}")
axes[1].imshow(binary_inverted, cmap="gray")


# draw blobs
for blob in blobs_log:
    y, x, r = blob
    circle = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
    axes[1].add_patch(circle)

plt.tight_layout()
plt.show()

