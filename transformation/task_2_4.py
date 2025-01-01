#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 22:07:12 2024

@author: asutimur
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# helper funct - create output file names
def get_path_with_new_filename_prefix(pathstr, new_prefix):
    p = Path(pathstr)
    p_name = p.name
    p = p.with_name(f'{new_prefix}{p_name}')
    return str(p)

# ---- PARAMETERS ---- #
img_path = 'vogel123.jpg'  
area_lower_bound = 2     # for smaller dots (birds)
area_upper_bound = 300   # max area to exclude larger objects
crop_left_fraction = 0.3  # crop out leftmost part to eliminate tree branches
circularity_threshold = 0.2  # looser circularity threshold for small dots

# ---- load img and convert to grayscale ---- #
print(f'Reading file...')
image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ----  crop ROI (Remove Left Side) ---- #
height, width = gray.shape
crop_x = int(width * crop_left_fraction)
cropped_gray = gray[:, crop_x:]  # Remove the leftmost 30%

# ----  reduce noise ---- #
blurred = cv2.GaussianBlur(cropped_gray, (5, 5), 0)

# ---- adaptive thresholding ---- #
print(f'Thresholding...')
threshed = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3
)

# ---- find kante ---- #
print(f'Finding contours...')
contours = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
print(f'Found {len(contours)} potential spots')

# ---- filter kante by area and circularity ---- #
found = 0
bird_mask = np.zeros_like(threshed)
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0:  # ! division by zero
        continue
    
    circularity = (4 * np.pi * area) / (perimeter ** 2)  # circularity formula
    
    # Looser circularity and size filter
    if area_lower_bound <= area <= area_upper_bound and circularity > circularity_threshold:
        found += 1
        cv2.drawContours(bird_mask, [contour], -1, 255, -1)  # fill detected birds

# ---- draw kante on origin img ---- #
output_image = image[:, crop_x:].copy()
bird_contours, _ = cv2.findContours(bird_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in bird_contours:
    color = random.choice([
        (0, 220, 220), (120, 255, 255), (220, 220, 0), (255, 255, 120),
        (220, 0, 220), (255, 120, 255), (120, 255, 120), (120, 120, 255)
    ])
    cv2.drawContours(output_image, [contour], -1, color, 2)

output_path = get_path_with_new_filename_prefix(img_path, f'birds-refined-{found}-')
print(f'Writing file to {output_path}...')
cv2.imwrite(output_path, output_image)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("original image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("cropped image (Sky Region)")
plt.imshow(cropped_gray, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("filtered bird mask")
plt.imshow(bird_mask, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title(f"birds detected: {found}")
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"bird count: {found}")
