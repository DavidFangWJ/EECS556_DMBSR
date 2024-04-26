# Evaluation of model in SSIM or LPIPS
import torch
import os
import tqdm
import lpips
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
                    , "init_upsample_mode": "nearest"},
           "is_train": False}

path = r'model_zoo/bicubic.pth'
netG = define_G(opt_net)
netG.load_state_dict(torch.load(path))
netG = netG.to('cuda')

# By my trials, the batch here is useless, and I don't know how to fix that
opt_data = { "phase": "train"
          , "dataloader_batch_size": 8
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
loss_fn_alex = lpips.LPIPS(net='alex').to('cuda')

scores = []

for sample in tqdm.tqdm(data):
    # Modification is needed because LPIPS package is based on PyTorch as opposed to NumPy
    HR = sample['H'][None].to('cuda')
    y = sample['L'][None].to('cuda')
    kmap = sample['kmap'][None].to('cuda')
    basis = sample['basis'][None].to('cuda')
    sf = sample['sf']
    sigma = sample['sigma'][None].to('cuda')

    res = netG(y, kmap, basis, sf, sigma)

    sc = loss_fn_alex(HR, res)
    scores.append(sc.item())

print(np.mean(scores))