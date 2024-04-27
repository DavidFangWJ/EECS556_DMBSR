import torch
from torch import nn
from models.model_plain import ModelPlain  # Adjust the import statement if necessary

import os
import random
import numpy as np

from scipy import ndimage
from scipy.io import loadmat

import torch
import torch.utils.data as data

from pycocotools.coco import COCO

import utils.utils_image as util
import utils.utils_sisr as sisr
from utils import utils_deblur

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import json

# Assuming necessary custom modules are in the path
from data.select_dataset import define_Dataset
from models.networks import define_G  # You need to implement or define this function based on your network structure

# Correctly structured configuration for ModelPlain
opt = {
    'num_channels': 3,  # Adjust this to match the model's requirements
    'upscale_factor': 2,
    'path': {
        'models': 'model_plain'  # Adjust this to the correct path where models are to be saved or accessed
    },
    'gpu_ids': None,  # Use None for CPU or specify GPU IDs, e.g., [0]
    'is_train': True,  # Set this to True if training the model, and False if you're running inference
    'train': {
        # Include relevant training parameters here, e.g.,
        'batch_size': 16,
        'learning_rate': 0.01,
        'scheduler_gamma': 0.1,
        'epochs': 100,
        # Any other training-related configurations
    },
    'netG': {
        # Configuration parameters for the generator network
        'type': None,  # Specify the type of model or architecture
        'num_features': 64,         # Example parameter, adjust based on actual model requirements
        'growth_rate': 32,          # Example for certain network types, e.g., DenseNets
        'net_type': None
        # Other necessary parameters specific to your network architecture
    }
}

# Load the pretrained model
model_path = 'model_zoo/DMBSR.pth'
pretrained_model = ModelPlain(opt)  # Pass the corrected configuration
pretrained_model.load_state_dict(torch.load(model_path))
pretrained_model.eval()  # Set the model to evaluation mode

class ModifiedModel(nn.Module):
    def __init__(self, pretrained_model, upscale_factor=2):
        super(ModifiedModel, self).__init__()
        # Assuming feature extraction layers are correctly extracted from pretrained_model
        self.feature_extractor = nn.Sequential(*list(pretrained_model.children())[:-1])
        num_channels = 64  # This should match the actual model's last layer output
        self.last_conv = nn.Conv2d(num_channels, num_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.sub_pixel = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.last_conv(x)
        x = self.sub_pixel(x)
        return x

# Initialize your modified model
modified_model = ModifiedModel(pretrained_model)



class Dataset(data.Dataset):
    def __init__(self, opt, use_subset=False, subset_size=1000):
        super(Dataset, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = self.opt['H_size']
        self.sigma = opt['sigma'] if opt['sigma'] else [0, 25]
        self.sigma_min, self.sigma_max = self.sigma[0], self.sigma[1]
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else 0
        self.scales = opt['scales'] if opt['scales'] is not None else [1, 2, 3, 4]
        self.motion_ker = loadmat('kernels/custom_blur_centered.mat')['kernels'][0]

        self.ksize = 33  # kernel size
        self.pca_size = 15
        self.min_p_mask = 20 ** 2  # Minimum number of pixels per mask to blur
        self.dataroot_H = self.opt['dataroot_H']
        self.coco = COCO(self.opt['coco_annotation_path'])
        indexes = self.coco.getImgIds()

        self.ids = []

        for i in indexes:
            img = self.coco.loadImgs(i)[0]
            if min(img['height'], img['width']) > self.opt['H_size'] + 49:
                self.ids.append(i)

        if use_subset:
            random.shuffle(self.ids)  # Shuffle to ensure random selection
            self.ids = self.ids[:subset_size]  # Limit the number of images to subset_size

        self.count = 0

    def __getitem__(self, index):
        # Existing method implementation
        pass

    def __len__(self):
        return len(self.ids)


from data.dataset_multiblur import Dataset

# Training configuration parameters
config = {
    "n_channels": 3,
    "H_size": 256,
    "sigma": [0, 25],
    "sigma_test": 5,
    "scales": [1, 2, 3, 4],
    "coco_annotation_path": "path/to/annotations.json",
    "dataroot_H": "path/to/dataroot_H"
}

# Initialize dataset with subsampling
use_subset = True
subset_size = 500  # Define how many images you want to use for training
train_dataset = Dataset(config, use_subset=use_subset, subset_size=subset_size)

# DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)


# Load training configuration
config_path = 'options/train_nimbusr.json'
with open(config_path, 'r') as f:
    config = json.load(f)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
train_set = define_Dataset(config['datasets']['train'])
train_loader = DataLoader(train_set, batch_size=config['datasets']['train']['dataloader_batch_size'], shuffle=True, num_workers=config['datasets']['train']['dataloader_num_workers'])

# Model
model = define_G(config['netG']).to(device)
if config['path'].get('pretrained_netG'):
    model.load_state_dict(torch.load(config['path']['pretrained_netG']))

# Loss and Optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config['train']['G_optimizer_lr'], weight_decay=config['train']['G_optimizer_wd'])
scheduler = MultiStepLR(optimizer, milestones=config['train']['G_scheduler_milestones'], gamma=config['train']['G_scheduler_gamma'])

# Training Loop
def train():
    model.train()
    for epoch in range(config['train']['n_epochs']):
        for i, data in enumerate(train_loader):
            L = data['L'].to(device)
            H = data['H'].to(device)

            # Forward pass
            outputs = model(L)
            loss = criterion(outputs, H)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Corrected print statement with proper handling of quotes
            if (i + 1) % config['train']['checkpoint_print'] == 0:
                print(f"Epoch [{epoch + 1}/{config['train']['n_epochs']}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}")

        scheduler.step()

        if (epoch + 1) % config['train']['checkpoint_save'] == 0:
            torch.save(model.state_dict(), os.path.join(config['path']['root'], f'model_epoch_{epoch + 1}.pth'))

if __name__ == '__main__':
    train()
