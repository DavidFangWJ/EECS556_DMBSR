import torch
import os
# import tqdm
import matplotlib.pyplot as plt

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
                    , "downsample_mode": "strideconv"},
           "is_train": False}

path_pretrained = r'model_zoo/DMBSR.pth'
netG_pretrained = define_G(opt_net)
netG_pretrained.load_state_dict(torch.load(path_pretrained))
netG_pretrained = netG_pretrained.to('cuda')

path_ours = r'model_zoo/recon_1.pth'
netG_ours = define_G(opt_net)
netG_ours.load_state_dict(torch.load(path_ours))
netG_ours = netG_ours.to('cuda')

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

for i in range(5):
    data.sf = 1
    data.count = -1
    sample = data[i]
    HR = util.tensor2uint(sample['H'])
    LR = util.tensor2uint(sample['L'])
    y = sample['L'][None].to('cuda')
    kmap = sample['kmap'][None].to('cuda')
    basis = sample['basis'][None].to('cuda')
    sf = sample['sf']
    sigma = sample['sigma'][None].to('cuda')
    
    res_pretrained = netG_pretrained(y, kmap, basis, sf, sigma)
    res_pretrained = util.tensor2uint(res_pretrained)

    res_ours = netG_ours(y, kmap, basis, sf, sigma)
    res_ours = util.tensor2uint(res_ours)
    
    plt.clf()
    plt.figure(figsize=(12,3))
    plt.subplot(141)
    plt.imshow(LR)
    plt.title('LR')
    plt.axis('off')
    plt.subplot(142)
    plt.imshow(res_pretrained)
    plt.title('Pretrained Model')
    plt.axis('off')
    plt.subplot(143)
    plt.imshow(res_ours)
    plt.title('Our reconstruction')
    plt.axis('off')
    plt.subplot(144)
    plt.imshow(HR)
    plt.title('HR')
    plt.axis('off')
    plt.savefig('Result #%d.png' % (i + 1))