#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 23:02:51 2024

@author: gs
"""

import cv2 as cv
import matplotlib.pyplot as plt

# Bild laden
image_path = "fisch4.bmp"
image = cv.imread(image_path)

# Farbkanäle extrahieren
blue_channel = image[:, :, 0]
green_channel = image[:, :, 1]
red_channel = image[:, :, 2]


# Kanäle anzeigen
plt.figure(figsize=(10, 5))
plt.subplot(131)
plt.title("Blau-Kanal")
plt.imshow(blue_channel, cmap="gray")
plt.axis("off")

plt.subplot(132)
plt.title("Grün-Kanal")
plt.imshow(green_channel, cmap="gray")
plt.axis("off")

plt.subplot(133)
plt.title("Rot-Kanal")
plt.imshow(red_channel, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
