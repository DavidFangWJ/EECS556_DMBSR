g
# Elevating Method in Super-Resolution with Non-Uniform Blur

This repository is the official implementation of [Elevating Method in Super-Resolution with Non-Uniform Blur](https://www.overleaf.com/project/65d41d3f46d6269c0e74e919). 
The code for NPLS is added to *models/network_dmbsr.py*.

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

| Configurations         | With NPLS  | Without NPLS |
| ------------------ |---------------- | -------------- |
| Low   |     24.54dB         |      23.91dB       |
| High | 24.94dB | 25.18dB |




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
