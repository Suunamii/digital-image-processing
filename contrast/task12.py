import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('kontrast_ex.jpg', cv.IMREAD_GRAYSCALE)

# gamma correction
gamma = 1.2  # values > 1 will make  image brighter, values < 1 will make  darker
gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype='uint8')



plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(gamma_corrected, cmap='gray')


plt.show()
