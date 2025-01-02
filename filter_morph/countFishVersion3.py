#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Wed Dec 18 20:59:37 2024
# author: gs

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""
# Load the images
image_path = "fisch1.bmp"
image = cv.imread(image_path)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

_, binary2 = cv.threshold(gray, 20, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
medianF = cv.medianBlur(binary2, 5)
_, binary = cv.threshold(gray, 50, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
cleaned = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)

contours, _ = cv.findContours(cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Draw contours and count fish
output_image = image.copy()
cv.drawContours(output_image, contours, -1, (0, 255, 0), 2)
fish_count = len(contours)
"""



source = "fisch4.bmp"
img_o = cv.imread(source)
if img_o is None:
    raise FileNotFoundError("Where is your image?!")

#  channel: green
ch_green = img_o[:, :, 1]

#  adapt thresholding 
img_b = cv.adaptiveThreshold(ch_green, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv.THRESH_BINARY_INV, 129, 12)

#  vanish noise - with morpho peration
kernel = np.ones((5, 7), np.uint8)

#img_b = cv.morphologyEx(img_b, cv.MORPH_CLOSE, kernel, iterations=3)
#img_b = cv.morphologyEx(img_b, cv.MORPH_OPEN, kernel, iterations=7)
medianF = img_b

#img_b= cv.medianBlur(medianF, 1)

img_b = cv.morphologyEx(img_b, cv.MORPH_OPEN, kernel, iterations=4)

img_b = cv.morphologyEx(img_b, cv.MORPH_CLOSE, kernel, iterations=4)
# ----  Konturen finden und filtern ---- #
contours, _ = cv2.findContours(img_b, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
output_image = img_b
cv2.drawContours(output_image, contours, -1, (0, 255, 0), 1)

media3= cv.medianBlur(output_image, 7)
kernel2 = np.ones((2, 5), np.uint8)
media4 = cv.morphologyEx(media3, cv.MORPH_OPEN, kernel2, iterations=3)

filtered_contours = []
for cnt in contours:
    area = cv.contourArea(cnt)
    if 500 <= area <= 8000:
        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = w / float(h)  
        if 1.5 <= aspect_ratio <= 4.0:  # fish contour (längliche Objekte)
            filtered_contours.append(cnt)

# count fish 
img_out = img_o.copy()
for contour in filtered_contours:
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(img_out, (x, y), (x + w, y + h), (0, 255, 0), 5)
    cv.ellipse(img_out, box, color)

fish_count = len(filtered_contours)
print(f"We found: {fish_count}")

# ---- creating image 
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title("Original")
#plt.imshow(cv.cvtColor(img_o, cv.COLOR_BGR2RGB))
plt.imshow(media4, cmap="gray")
plt.axis("off")

plt.subplot(122)
plt.title(f"processed img -- How many Fish? -> {fish_count}")
plt.imshow(img_b, cmap="gray")
plt.axis("off")
plt.show()
