#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 21:17:03 2024

@author: gs
----------------------------------------------------------------

"""
"""
Bildaufteilung:

Das Bild wird in kleine Bereiche (lokale Blöcke) unterteilt.
Für jeden Block wird ein eigener Schwellwert berechnet.
Schwellwert-Berechnung:

Der Schwellwert wird entweder als Mittelwert (Mean) oder als gewichteter Mittelwert (Gaussian) der Pixelintensitäten im Block berechnet.
Feinabstimmung mit der Konstante C:

Nachdem der Schwellwert für den Block berechnet wurde, wird eine Konstante C davon abgezogen.
Mit C kann man das Ergebnis feinjustieren.

Blockgröße (blockSize):

Eine kleine Blockgröße (z.B. 15) berechnet den Schwellwert aus kleinen Nachbarbereichen:
Vorteil: Gut für Bereiche mit schnellen Intensitätswechseln.
Nachteil: Kann zu Rauschen führen.
Eine große Blockgröße (z.B. 35) berücksichtigt mehr Nachbarpixel:
VorteGlatteres Ergebnis und weniger Rauschen.
Nachteil: Details können verloren gehen.
Konstante C:

C justiert den Schwellwert, um das Ergebnis zu optimieren:
Ein positives C (z.B. 5) macht die Schwelle strenger → weniger weiße Pixel.
Ein negatives C (z.B. -5) macht die Schwelle großzügiger → mehr weiße Pixel.
"""

import cv2
import matplotlib.pyplot as plt

image = cv2.imread('rice.png', cv2.IMREAD_GRAYSCALE)

manual_thresh_value = 127

_, manual_thresh = cv2.threshold(image, manual_thresh_value, 255, cv2.THRESH_BINARY)

_, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 45, 2)

"""
Adaptive thresholding is a method used in image segmentation 
to convert a grayscale image into a binary image (black and white). 
Unlike global thresholding (where one threshold value is applied to the entire image), adaptive thresholding calculates the threshold locally for small regions of the image.
This is particularly useful for images with uneven lighting 
or shadows.

"""
# Step 5: Plot the results
plt.figure(figsize=(10, 8))


plt.subplot(221)
plt.title("Originalbild")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(222)
plt.title(f"Manuelles Thresholding (Wert = {manual_thresh_value})")
plt.imshow(manual_thresh, cmap='gray')
plt.axis('off')

plt.subplot(223)
plt.title("Otsu's Thresholding")
plt.imshow(otsu_thresh, cmap='gray')
plt.axis('off')

plt.subplot(224)
plt.title("Adaptive Thresholding")
plt.imshow(adaptive_thresh, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()


