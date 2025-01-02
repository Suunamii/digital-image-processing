#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 23:13:04 2024

@author: gs
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


image_path = "fisch4.bmp"  
original_img = cv.imread(image_path)
if original_img is None:
    raise FileNotFoundError("Bild nicht gefunden. Pfad prüfen!")

blue_channel = original_img[:, :, 0] 
red_channel = original_img[:, :, 0]   

# ---- channel sub ---- #
combined_channel = cv.subtract(blue_channel, red_channel)

# ---- trying stuff ---- #
_, binary_img = cv.threshold(combined_channel, 50, 255, cv.THRESH_BINARY)

# ---- morph operation ---- #
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
binary_img = cv.morphologyEx(binary_img, cv.MORPH_CLOSE, kernel, iterations=2)
binary_img = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel, iterations=2)

# ---- find contour ---- #
contours, _ = cv.findContours(binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


min_area, max_area = 900, 7000  # Nur große Objekte zählen
filtered_contours = [cnt for cnt in contours if min_area <= cv.contourArea(cnt) <= max_area]

# ---- trying stuff ---- #
output_img = original_img.copy()
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

fish_count = len(filtered_contours)
print(f"Anzahl der Fische: {fish_count}")

# ---- 8. Ergebnisse anzeigen ---- #
plt.figure(figsize=(12, 6))

plt.subplot(131)
plt.title("Blau-Kanal")
plt.imshow(blue_channel, cmap="gray")
plt.axis("off")

plt.subplot(132)
plt.title("Rot-Kanal")
plt.imshow(red_channel, cmap="gray")
plt.axis("off")

plt.subplot(133)
plt.title(f"Binärbild zur Zählung (Fische: {fish_count})")
plt.imshow(binary_img, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
