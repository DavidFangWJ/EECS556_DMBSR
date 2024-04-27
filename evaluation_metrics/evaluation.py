'''
calculate the Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS) for assessing image quality, especially in tasks like super-resolution
'''
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import lpips
import torch

# import os
# import ssl

# # Make sure to verify the path to the cert.pem file
# os.environ['SSL_CERT_FILE'] = '/path/to/cert.pem'
# ssl._create_default_https_context = ssl._create_unverified_context
HR_image_path = 'images/Generalization/1_HR.png'
Compare_image_path = 'bilateral_filtered_image_blur.png'

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Unable to load image at {path}")
    return image

def psnr(target, ref):
    mse = np.mean((target - ref) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_ssim(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(img1_gray, img2_gray, full=True)
    return score

def calculate_lpips(img1, img2):
    img1_tensor = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2_tensor = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    lpips_net = lpips.LPIPS(net='alex')
    dist = lpips_net(img1_tensor, img2_tensor)
    return dist.item()

try:
    target_image = load_image(HR_image_path) # target
    ref_image = load_image(Compare_image_path) # reference

    # # Resize reference image to target image dimensions
    # ref_image = cv2.resize(ref_image, (target_image.shape[1], target_image.shape[0]))
    
    print("PSNR:", psnr(target_image, ref_image))
    print("SSIM:", calculate_ssim(target_image, ref_image))
    print("LPIPS:", calculate_lpips(target_image, ref_image))
except Exception as e:
    print(e)
