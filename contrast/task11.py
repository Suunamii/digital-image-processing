import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('kontrast_ex.jpg', cv.IMREAD_GRAYSCALE)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(img)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("CLAHE Enhanced Image")
plt.imshow(clahe_img, cmap='gray')
plt.show()
