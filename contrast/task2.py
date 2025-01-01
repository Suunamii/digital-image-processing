import cv2
import numpy as numpy
import matplotlib.pyplot as plt

image = cv2.imread('zellen_1.png') 
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # bgr to rgb convertion 

# enhance image quallity
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) 
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl = clahe.apply(l)
enhanced_lab = cv2.merge((cl, a, b))
enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

# identify cells , using edge detection and contours
gray  = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(blur, 50, 150) 

# find contours
cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# draw contours and measure areas
for contour in cnts:
    if len(contour) >= 5:
        area = cv2.contourArea(contour)
        if area > 50:
            cv2.drawContours(enhanced_image, [contour], -1, (0, 255, 0), 2)

# calc color histogram
colors = ('r', 'g', 'b')
plt.figure(figsize=(12, 6))
for i, col in enumerate(colors):
    hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

plt.title('Color Histogram for Red, Green, and Blue channels')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 10))
plt.title("Enhanced Image with Cell Contours")
plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()       
        


