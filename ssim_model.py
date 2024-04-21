# Evaluation of model in SSIM or LPIPS
import torch
import os
import tqdm
from skimage.metrics import structural_similarity as ssim
import numpy as np

from models.select_network import define_G
from data.dataset_multiblur import Dataset

import utils.utils_image as util
import utils.utils_sisr as sisr


DATA_DIR = 'datasets/COCO'

opt_net = {"netG" : {"net_type": "dmbsr"
                    , "n_iter": 8
                    , "h_nc": 64
                    , "in_nc": 4
                    , "out_nc": 3
                    , "ksize": 25
                    , "nc": [64, 128, 256, 512]
                    , "nb": 2
                    , "gc": 32
                    , "ng": 2
                    , "reduction" : 16
                    , "act_mode": "R" 
                    , "upsample_mode": "convtranspose" 
                    , "downsample_mode": "strideconv"
                    , "init_upsample_mode": "bilinear"},
           "is_train": False}

path = r'model_zoo/zimeng_1.pth'
netG = define_G(opt_net)
netG.load_state_dict(torch.load(path))
netG = netG.to('cuda')

opt_data = { "phase": "train"
          , "dataloader_batch_size": 1
          , "dataset_type": "multiblur"
          , "dataroot_H": os.path.join(DATA_DIR, 'val2014')
          , "H_size": 256
          , "scales": [2]
          , "sigma": [5, 10]
          , "sigma_test": 10
          , "n_channels": 3
          , "motion_blur": True
          , "coco_annotation_path": os.path.join(DATA_DIR, 'instances_val2014.json')}

data = Dataset(opt_data)

scores = []

for sample in tqdm.tqdm(data):
    HR = util.tensor2uint(sample['H'])
    LR = util.tensor2uint(sample['L'])
    y = sample['L'][None].to('cuda')
    kmap = sample['kmap'][None].to('cuda')
    basis = sample['basis'][None].to('cuda')
    sf = sample['sf']
    sigma = sample['sigma'][None].to('cuda')

    res = netG(y, kmap, basis, sf, sigma)
    res = util.tensor2uint(res)

    # May be replaced with LPIPS
    sc = ssim(HR, res, data_range=res.max() - res.min(), channel_axis=2)
    scores.append(sc)

print(np.mean(scores))