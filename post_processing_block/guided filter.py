import cv2
import os
import sys
print(sys.executable)

# Check OpenCV version
print(cv2.__version__)

# Try to access the ximgproc module
try:
    _ = cv2.ximgproc.guidedFilter
    print("ximgproc module is accessible.")
except AttributeError as e:
    print("ximgproc module is not accessible:", e)

# Print the current working directory
print("Current working directory:", os.getcwd())

# Define the absolute paths to your images
super_resolved_image = 'images/Generalization/1_nimbusr.png'
original_image = 'images/Generalization/1_HR.png'

# Load your super-resolved image
image = cv2.imread(super_resolved_image)
# Load the guidance image
guidance_image = cv2.imread(original_image)

# Check if images are loaded properly
if image is None or guidance_image is None:
    if image is None:
        print(f"Error: Unable to load the super-resolved image from {super_resolved_image}. Check the file path and integrity.")
    if guidance_image is None:
        print(f"Error: Unable to load the original image from {original_image}. Check the file path and integrity.")
else:
    # Apply guided filter
    filtered_image = cv2.ximgproc.guidedFilter(guide=guidance_image, src=image, radius=8, eps=100)

    # Save or display the result
    cv2.imwrite('guided_filtered_image_blur.png', filtered_image)
    print("Guided filtering completed and image saved.")


# import numpy as np

# def guided_filter(I, p, r, eps):
#     """
#     Perform guided filtering on an image.

#     Parameters:
#     I   : 2D or 3D array, guidance image (grayscale image or color image)
#     p   : 2D array, filtering input image (grayscale image)
#     r   : scalar, radius of the guidance
#     eps : scalar, regularization term

#     Returns:
#     q   : 2D array, filtering output image (grayscale image)
#     """
#     # Image dimensions
#     height, width = p.shape

#     # Number of elements in the window
#     N = (2*r + 1) ** 2

#     # Summed area table for fast area sum
#     def sum_box(img):
#         return cv2.boxFilter(img, -1, (r, r), borderType=cv2.BORDER_REFLECT)

#     # Mean of input and guidance images
#     mean_I = sum_box(I) / N
#     mean_p = sum_box(p) / N

#     # Mean of products and squared guidance
#     mean_Ip = sum_box(I * p) / N
#     mean_II = sum_box(I * I) / N

#     # Covariance and variance
#     cov_Ip = mean_Ip - mean_I * mean_p
#     var_I = mean_II - mean_I * mean_I

#     # Coefficients
#     a = cov_Ip / (var_I + eps)
#     b = mean_p - a * mean_I

#     # Mean coefficients
#     mean_a = sum_box(a) / N
#     mean_b = sum_box(b) / N

#     # Output image
#     q = mean_a * I + mean_b
#     return q

# # Example usage
# from skimage import data, img_as_float
# import matplotlib.pyplot as plt

# # Load an example image (grayscale for simplicity)
# image = img_as_float(data.camera())
# guidance = image  # self-guided
# radius = 5
# epsilon = 0.1**2

# # Apply the guided filter
# result = guided_filter(guidance, image, radius, epsilon)

# # Display the original and the result
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(image, cmap='gray')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title('Filtered Image')
# plt.imshow(result, cmap='gray')
# plt.axis('off')

# plt.show()


