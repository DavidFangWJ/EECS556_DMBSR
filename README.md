# Improvement Methods in Super-Resolution with Non-Uniform Blur

This repository is the improvement methods for official implementation of [Elevating Method in Super-Resolution with Non-Uniform Blur](https://www.overleaf.com/project/65d41d3f46d6269c0e74e919). 

Improvement methods include classical image post-processing and sub-pixel convolution w or w/o frozen training.

### Post-Processing Block with Bilateral Filter, Guided Filter Or Wavelet
The Bilateral Filter, Guided Filter and Wavelet Realizations are in *post_processing_block/* file.
![post_pic](https://github.com/DavidFangWJ/EECS556_DMBSR/assets/130185305/e3a985e4-d40b-4a20-863b-d9710d2473f3)

### Sub-pixel Convolution Processing and Frozen Training
The Sub-pixel Convolution Processing and Frozen Training Process are in *sub_pixel_conv/* file.
![sub_figure](https://github.com/DavidFangWJ/EECS556_DMBSR/assets/130185305/201c641f-1c1f-4b8d-bc9f-2b2aa635fd50)

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

See *evaluation_metrics/* flile for PSNR, SSM, LPIPS Realizations.
See *example_outputs/* file for example output images with these improvement methods.

## Pre-trained Models

The pretrained model of the original model is in *model_zoo/DMBSR.pth*.
For NPLS model, it is located in *SR/nimbusr/models/22500_G.pth*

## Results

Our model achieves the following performance on :

### [COCO Dataset](https://cocodataset.org)


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
