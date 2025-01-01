import cv2
import numpy as np

image = cv2.imread('meer.jpg')

resized_image = cv2.resize(image, (800,600))

cv2.imwrite('resized_meer.jpg', resized_image)


angle = -5
(h, w) = resized_image.shape[:2]
center = (w // 2, h // 2)

rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_image = cv2.warpAffine(resized_image, rotation_matrix, (w, h))

cv2.imwrite('rotated_meer.jpg', rotated_image)

cropped_image = rotated_image[50:550, 100:700]

cv2.imwrite('cropped_meer.jpg', cropped_image)

enlarged_image_nearest  = cv2.resize(cropped_image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
enlarged_image_linear = cv2.resize(cropped_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
enlarged_image_cubic = cv2.resize(cropped_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

cv2.imwrite('img_nearest.jpg', enlarged_image_nearest)
cv2.imwrite('img_linear.jpg', enlarged_image_linear)
cv2.imwrite('img_cubic.jpg', enlarged_image_cubic)

image_t = cv2.imread('resized_meer.jpg', cv2.IMREAD_GRAYSCALE)
_, binary_image = cv2.threshold(image_t, 100, 255, cv2.THRESH_BINARY)
cv2.imwrite('thres_img.jpg', binary_image)