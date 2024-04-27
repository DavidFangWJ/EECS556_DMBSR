import torch
from torch import nn
from models import model_plain  

# Load the pretrained model
model_path = 'model_zoo/DMBSR.pth'  # Path to the pretrained model
pretrained_model = model_plain()
pretrained_model.load_state_dict(torch.load(model_path))
pretrained_model.eval()  # Set the model to evaluation mode if not training right away

class ModifiedModel(nn.Module):
    def __init__(self, pretrained_model, upscale_factor=2):
        super(ModifiedModel, self).__init__()
        # Assume all layers except the last are used
        self.feature_extractor = nn.Sequential(*list(pretrained_model.children())[:-1])  
        
        # Last convolution layer of pretrained model before upscaling
        # Here you need to know the number of output channels from the last layer
        num_channels = 64  # Example value, adjust according to your model's last layer output
        self.last_conv = nn.Conv2d(num_channels, num_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        
        # Sub-pixel convolution layer
        self.sub_pixel = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.last_conv(x)
        x = self.sub_pixel(x)
        return x

# Initialize your modified model
modified_model = ModifiedModel(pretrained_model)