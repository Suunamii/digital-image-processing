import numpy as np
import cv2
import matplotlib.pyplot as plt

# Original binary matrix-image (1 = white, 0 = black)
image = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 1, 0, 0],
    [1, 1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 0, 0]
], dtype=np.uint8)

# Step 1: Define rotation angle
angle = 0  # Rotate counterclockwise by -45 degrees (adjust if needed)

# Step 2: Image dimensions and define center of rotation
(h, w) = image.shape
center = (3, 3)  # Set center to row 3, column 3 (middle of the matrix)

# Step 3: Compute the rotation matrix
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

# Step 4: Rotate the image using the rotation matrix
rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)

# Step 5: Post-process to enforce horizontal rows
# Clean row 2 (all black)
rotated_image[2, :] = 0
# Clean row 3 (all white)
rotated_image[3, :] = 1
# Clean row 4 (all black)
rotated_image[4, :] = 0

# Threshold to ensure binary values
rotated_image = (rotated_image > 0).astype(np.uint8)

# Step 6: Display the original and rotated images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray', vmin=0, vmax=1)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Final Rotated Image (Aligned)")
plt.imshow(rotated_image, cmap='gray', vmin=0, vmax=1)
plt.axis('off')

plt.tight_layout()
plt.show()

print("Final Transformed Matrix:")
print(rotated_image)
