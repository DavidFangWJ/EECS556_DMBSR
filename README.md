# Elevating Method in Super-Resolution with Non-Uniform Blur

This repository is the official implementation of [Elevating Method in Super-Resolution with Non-Uniform Blur](https://www.overleaf.com/project/65d41d3f46d6269c0e74e919). 

The code for training with different sizes of training subset is achieved by modifying `main_train.py`.

The change in the number of iterations is done by modifying training JSON file.

The evaluation of SSIM and LPIPS of the models, which is not given in the training step, is found at `ssim_model.py` and `lpips_model.py`.

The switch to alternative upsampling method is implemented at `network_dmbsr.py` and may be selected from training JSON file.

## Requirements

To install requirements:

```setup
conda create -n dmbsr python=3.6
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install tqdm
conda install matplotlib
pip install opencv-python
conda install conda-forge/label/main::pycocotools
conda install scipy
```
The blur kernels are available for download [here](https://drive.google.com/file/d/1o1ruvDSbR9R12DzjA-2KIps7cqy4544v/view?usp=share_link). They need to be added in the folder |-*kernels*

These requirements should be located in conda_env.txt. If there is still any package missing, please install using conda or pip.

## Training

To train the model(s) in the paper, please first download COCO dataset available at: https://cocodataset.org

```train
python main_train.py -opt options/train_nimbusr.json
python mytrain.py -opt options/train_nimbusr.json # mytrain.py is a modified version for our own training
```


## Evaluation

See *test_model.ipynb* to test the model on COCO dataset.
See *results/* folder for plotting functions.

## Pre-trained Models

The pretrained model of the original model is in *model_zoo/DMBSR.pth*.
For NPLS model, it is located in *SR/nimbusr/models/22500_G.pth*

## Results

Our model achieves the following performance on :

### [COCO Dataset](https://cocodataset.org)

| Upsampling method | PSNR (dB) | SSIM | LPIPS |
| ----------------- | --------- | ---- | ----- |
| Original (nearest)} | 24.71   | 0.66 | 0.35  |
| Bilinear | 24.75 | 0.64 | 0.38 |
| Bicubic  | 24.68 | 0.63 | 0.40 |
| (Original Paper) | 25.36 | 0.73 | 0.28 |
| (Average state of the art) | 23.11 | 0.64 | 0.43 |

## Contributing

We are working on the code based on the following work.
```
@InProceedings{laroche2023dmbsr,
  title = {Deep Model-Based Super-Resolution with Non-Uniform Blur},
  author = {Laroche, Charles and Almansa, Andrés and Tassano, Matias},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}
  year = {2023}
}
```
