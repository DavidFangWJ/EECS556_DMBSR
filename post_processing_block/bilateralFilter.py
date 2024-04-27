import cv2
import numpy as np

# Assuming 'super_resolved_image.jpg' is the output of your existing super-resolution process
image = cv2.imread('images/Generalization/1_nimbusr.png')

# Apply bilateral filter
# d: Diameter of each pixel neighborhood
# sigmaColor: Value of \(\sigma\) in the color space. The greater the value, the colors farther to each other will start to get mixed.
# sigmaSpace: Value of \(\sigma\) in coordinate space. The greater its value, the more distant pixels will influence each other.
filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

# Optionally, save or display the resulting image
cv2.imwrite('bilateral_filtered_image_blur.png', filtered_image)