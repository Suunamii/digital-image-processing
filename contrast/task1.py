import cv2 as cv
import matplotlib.pyplot as plt

#load image
img = cv.imread('kontrast_ex.jpg', cv.IMREAD_GRAYSCALE)

equalized_image = cv.equalizeHist(img)

#plot histogram
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.title("Original Image Histogram")
plt.hist(img.ravel(), bins=256, range=(0, 256))
plt.subplot(122)
plt.title("Enhanced Image Histogram")
plt.hist(equalized_image.ravel(), bins=256, range=(0, 256))
plt.show()
