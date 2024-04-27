########## unstable color image ###########
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image



class SubPixelRefinement(nn.Module):
    def __init__(self, input_channels=3, upscale_factor=2):
        super(SubPixelRefinement, self).__init__()
        # Create a Conv2D layer that increases the number of channels by upscale_factor^2
        self.refine_conv = nn.Conv2d(input_channels, input_channels * upscale_factor**2,
                                     kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
    
    def forward(self, x):
        x = self.refine_conv(x)
        x = self.pixel_shuffle(x)
        return x

# Initialize the model
refinement_module = SubPixelRefinement(input_channels=3, upscale_factor=2)
refinement_module.eval()  # Set the model to evaluation mode if there are any layers specific to training

# Load an image
image_path = 'images/Generalization/1_LR.png'  # Make sure to provide the correct path to your image file
image = Image.open(image_path).convert('RGB')  # Convert image to RGB if not already

# Prepare the image
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Resize((256, 256))  # Resize the image if necessary
])
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Process the image through the model
refined_image = refinement_module(input_tensor)

# Convert the processed tensor to an image for visualization
output_image = refined_image.squeeze(0).detach()  # Remove the batch dimension and detach from the graph
output_image = output_image.permute(1, 2, 0)  # Change from [C, H, W] to [H, W, C] for Matplotlib
output_image = output_image.numpy()  # Convert to numpy array for display

# Display the image
plt.imshow(output_image)
plt.title("Upscaled Image")
plt.axis('off')  # Hide axes
plt.show()

# Save the image
output_pil = to_pil_image(refined_image.squeeze(0))  # Convert tensor directly to PIL Image
output_pil.save('upscaled_image3.png')  # Save the image to disk

# # Save the figure
# plt.imshow(output_image)
# plt.axis('off')  # Ensure axes are still off
# plt.savefig('upscaled_image2.png', bbox_inches='tight', pad_inches=0)
# plt.close()  # Close the plot to free up memory

# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from PIL import Image
# import matplotlib.pyplot as plt
# from torchvision.transforms.functional import to_pil_image
# import numpy as np
# import cv2


# def match_histograms(source, reference):
#     """
#     Adjust the pixel values of a grayscale image such that its histogram
#     matches that of a target image
#     """
#     oldshape = source.shape
#     source = source.ravel()
#     reference = reference.ravel()

#     # Get the set of unique pixel values and their corresponding indices and counts
#     s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
#     r_values, r_counts = np.unique(reference, return_counts=True)

#     # Calculate cumulative distribution function for both images
#     s_quantiles = np.cumsum(s_counts).astype(np.float64)
#     s_quantiles /= s_quantiles[-1]
#     r_quantiles = np.cumsum(r_counts).astype(np.float64)
#     r_quantiles /= r_quantiles[-1]

#     # Interpolate pixel values in source image to match the histogram of the target image
#     interp_t_values = np.interp(s_quantiles, r_quantiles, r_values)
#     return interp_t_values[bin_idx].reshape(oldshape)



# from scipy.ndimage import gaussian_filter

# def reduce_noise(image, sigma=0.5):
#     return gaussian_filter(image, sigma=sigma)

# ################

# def denoise_image(image):
#     """
#     Apply OpenCV's non-local means denoising function to remove noise.
#     """
#     return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# def convert_colorspace(image, conversion_code):
#     """
#     Convert the image color space using OpenCV.
#     """
#     return cv2.cvtColor(image, conversion_code)


# class SubPixelRefinement(nn.Module):
#     def __init__(self, input_channels=3, upscale_factor=2):
#         super(SubPixelRefinement, self).__init__()
#         self.refine_conv = nn.Conv2d(input_channels, input_channels * upscale_factor**2,
#                                      kernel_size=3, padding=1)
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
    
#     def forward(self, x):
#         x = self.refine_conv(x)
#         x = self.pixel_shuffle(x)
#         return x



# # Initialize the model
# refinement_module = SubPixelRefinement(input_channels=3, upscale_factor=2)
# refinement_module.eval()

# # Load and prepare the image
# image_path = 'images/Generalization/1_LR.png'
# image = Image.open(image_path).convert('RGB')
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((256, 256))  # Assuming you want to resize before input to model
# ])
# input_tensor = transform(image).unsqueeze(0)

# # Process the image through the model
# refined_image = refinement_module(input_tensor)

# # Convert the processed tensor back to an image
# output_image = refined_image.squeeze(0).detach().permute(1, 2, 0).numpy()

# # Prepare for histogram matching
# original_image = np.array(image.resize((512, 512))) / 255.0  # Resizing to match the processed image size

# # Apply histogram matching and noise reduction
# matched_image = match_histograms(output_image, original_image)
# denoised_image = reduce_noise(matched_image, sigma=0.5)

# # Convert denoised image to the correct format for OpenCV
# denoised_image = (denoised_image * 255).astype(np.uint8)  # Scale and convert to uint8
# rgb_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)  # Assuming the denoised image is in BGR format

# # Display the final processed image
# plt.imshow(rgb_image)
# plt.title("Upscaled and Color Corrected Image")
# plt.axis('off')
# plt.show()






