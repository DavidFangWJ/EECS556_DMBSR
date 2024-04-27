import pywt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(file_path):
    with Image.open(file_path) as img:
        return np.array(img)

def wavelet_super_resolution_channel(channel, wavelet='haar', levels=2):
    # Decompose the channel into wavelet coefficients
    coeffs = pywt.wavedec2(channel, wavelet, level=levels)
    coeffs_new = list(coeffs)
    
    # Enhance the approximation coefficients slightly (this affects the overall brightness)
    coeffs_new[0] *= 1.1  # Smaller factor to prevent fading

    # Enhance the detail coefficients (this affects the sharpness)
    for i in range(1, len(coeffs)):
        coeffs_new[i] = tuple([np.clip(c * (1 + 0.02 * (i - 1)), 0, 255) for c in coeffs[i]])

    # Reconstruct the channel from the new coefficients
    return pywt.waverec2(coeffs_new, wavelet)

def wavelet_super_resolution_color(image, wavelet='haar', levels=2):
    # Ensure that the image is in float64 format to prevent clipping during operations
    image = image.astype(np.float64)

    # Process each color channel independently
    channels = [wavelet_super_resolution_channel(image[:,:,i], wavelet, levels) for i in range(3)]
    
    # Stack the channels back together and clip to valid range [0, 255]
    enhanced_image = np.stack(channels, axis=-1)
    enhanced_image = np.clip(enhanced_image, 0, 255)

    # Convert the image back to uint8 format
    return enhanced_image.astype(np.uint8)

# Load the image
image_path = 'images/Generalization/1_LR.png'
image = load_image(image_path)

# Enhance the image
enhanced_image = wavelet_super_resolution_color(image)
# Convert the enhanced image back to a PIL image for saving
enhanced_image_pil = Image.fromarray(np.uint8(enhanced_image))

# Save the enhanced image
output_path = 'enhanced_image2.png'
enhanced_image_pil.save(output_path)

# Display the original and enhanced images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(enhanced_image, cmap='gray')
plt.title('Enhanced Image')
plt.axis('off')

plt.show()



# import numpy as np

# def haar_1d(signal):
#     # Assure signal length is even
#     assert len(signal) % 2 == 0, "The length of the signal must be even."
    
#     avg = (signal[::2] + signal[1::2]) / 2.0
#     diff = (signal[::2] - signal[1::2]) / 2.0
    
#     return avg, diff

# def ihaar_1d(avg, diff):
#     signal = np.empty((avg.size + diff.size,), dtype=avg.dtype)
#     signal[::2] = avg + diff
#     signal[1::2] = avg - diff
    
#     return signal

# def haar_2d(image):
#     # Perform the Haar transform on rows:
#     row_trans = np.array([haar_1d(row) for row in image])
#     avg_rows = row_trans[:,0,:]
#     diff_rows = row_trans[:,1,:]

#     # Perform the Haar transform on columns:
#     col_trans = np.array([haar_1d(col) for col in avg_rows.T])
#     avg = col_trans[:,0,:].T
#     detail_avg = col_trans[:,1,:].T

#     col_trans = np.array([haar_1d(col) for col in diff_rows.T])
#     detail_diff = col_trans[:,0,:].T
#     diff = col_trans[:,1,:].T

#     # Combine the averages and details:
#     top = np.hstack((avg, detail_avg))
#     bottom = np.hstack((detail_diff, diff))
#     result = np.vstack((top, bottom))
    
#     return result

# def ihaar_2d(coefficients):
#     # Split the image into four quarters:
#     N, M = coefficients.shape
#     N2, M2 = N // 2, M // 2
    
#     avg = coefficients[:N2, :M2]
#     detail_avg = coefficients[:N2, M2:]
#     detail_diff = coefficients[N2:, :M2]
#     diff = coefficients[N2:, M2:]
    
#     # Inverse Haar transform on columns:
#     col_avg = np.array([ihaar_1d(avg[:,i], detail_avg[:,i]) for i in range(M2)]).T
#     col_diff = np.array([ihaar_1d(detail_diff[:,i], diff[:,i]) for i in range(M2)]).T
    
#     # Inverse Haar transform on rows:
#     result = np.array([ihaar_1d(col_avg[i,:], col_diff[i,:]) for i in range(N2)])
    
#     return result

# # Example usage:
# image = np.random.rand(4, 4)  # Example image
# coefficients = haar_2d(image)
# reconstructed_image = ihaar_2d(coefficients)

# # Check if the original and reconstructed images are the same
# print(np.allclose(image, reconstructed_image))