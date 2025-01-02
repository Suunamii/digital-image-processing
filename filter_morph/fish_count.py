#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:02:57 2024

@author: gs
"""

"""
a)

Step 1.: Bildvorbereitung
        - Graustufenbild konventieren
        - Rausch reduzieren (e.g. Gaussian blur)
Step 2.: Hintergrungsubtraktion
        - statischer Hintergrund entfernen
Step 3.: Thresholding
        - binary convertion
        - otsu / adaptive ?
Step 4.: Morphologic Operation
        - Dilation : connect disturbed contours
        - Erosion 
Step 5.: Count Fish / Contour
Step 6.: Visualize

b)

Rot-Grün-und Blau-Kanal = Graustufenbild
ohne Grün und oder Blau Kanal gehen Informationen verloren.
Standardformel zur Grauwertbildung:

Grauwert = 0.299⋅R + 0.587⋅G + 0.114⋅B

"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


source = "fisch1.bmp"
img_o = cv.imread(source)
if img_o is None:
    raise FileNotFoundError("Where is your image?!")

#  channel: green
ch_green = img_o[:, :, 1]

#  adapt thresholding 
img_b = cv.adaptiveThreshold(ch_green, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv.THRESH_BINARY_INV, 87, 10)

#  vanish noise - with morpho peration
kernel = np.ones((1, 2), np.uint8)

img_b = cv.morphologyEx(img_b, cv.MORPH_CLOSE, kernel, iterations=3)
img_b = cv.morphologyEx(img_b, cv.MORPH_OPEN, kernel, iterations=7)



# ----  Konturen finden und filtern ---- #
contours, _ = cv.findContours(img_b, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

filtered_contours = []
for cnt in contours:
    area = cv.contourArea(cnt)
    if 1000 <= area <= 8000:
        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = w / float(h)  # Seitenverhältnis prüfen
        if 1.5 <= aspect_ratio <= 4.0:  # Fischform (längliche Objekte)
            filtered_contours.append(cnt)

# count fish 
img_out = img_o.copy()
for contour in filtered_contours:
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(img_out, (x, y), (x + w, y + h), (0, 255, 0), 2)

fish_count = len(filtered_contours)
print(f"We found: {fish_count}")

# ---- creating image 
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title("Original")
plt.imshow(cv.cvtColor(img_o, cv.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(122)
plt.title(f"processed img -- How many Fish? -> {fish_count}")
plt.imshow(img_b, cmap="gray")
plt.axis("off")
plt.show()









