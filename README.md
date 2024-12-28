# Neural Film Grain Rendering

Official implementation of the paper [*Neural Film Grain Rendering*](https://hal.science/hal-04667141)

![Python 3.10](https://img.shields.io/badge/Python-3.10-yellow.svg)
![pytorch 2.5.0](https://img.shields.io/badge/Pytorch-2.5.0-blue.svg)

![image](./images/teaser.png)
**Figure:** *Film Grain rendered by our method*

> **Neural Film Grain Rendering** <br>
>  Gwilherm Lesné, Yann Gousseau, Saïd Ladjal, Alasdair Newson <br>
>  LTCI, Télécom Paris <br>
>  ISIR, Sorbonne Université <br>

## Set up

Clone this repository
```bash
git clone https://github.com/Gwilherm-LESNE/Neural_Film_Grain_Rendering.git
cd Neural_Film_Grain_Rendering/
```

Install the required libraries
```bash
conda env create -f filmgrain.yml
```

Load the conda environment
```bash
conda activate filmgrain
```
**N.B.** This code relies on the official CUDA implementation of [**A Stochastic Film Grain Model for Resolution‐Independent Rendering**](https://onlinelibrary.wiley.com/doi/10.1111/cgf.13159) for creating the database and computing the metrics. Please follow the **requirements** [here](https://github.com/alasdairnewson/film_grain_rendering_gpu) to use it.

## Getting data

You have two options: 
- You want to use the data we used :
  - Download the zip archive [here](https://www.kaggle.com/datasets/gwilhermlsn/neuralfilmgrainrendering)
  - Unzip `NeuralFilmGrainRendering_Dataset.zip`
  - The training grainy images are in `Mixed_grain/train` and the training grain-free images are in `Mixed/train`
- You want to train the model on your own data:
  - Get the code [here](https://github.com/alasdairnewson/film_grain_rendering_gpu) and follow the steps to be able to run it properly
  - Add the path to the executable (film_grain_rendering_main) in `dataset.py`
  - Execute the following command and follow the instructions (do not forget to activate the environment before):
    ```
    python dataset.py -p ./path/to/grain-free/images/
    ```

## Training

### Training GrainNet

```bash
python train.py -i './input/path' -g './grain/path'
```
Required:
  - `-i` Path to the folder containing the **grain-free** images
  - `-g` Path to the folder containing the **grainy** images

Options:
  - `-s` Folder in which to save the model and its checkpoint
  - `-lr` Learning rate
  - `-bs` Batch size
  - `-ne` Number of epochs
  - `-ss` Step size for optimizer scheduler
  - `-ga` Gamma for optimizer scheduler
  - `-si` Indices where to take the activation map of VGG
  - `-sw` Weights for the corresponding activation maps of VGG
  - `-ac` Activation function to use
  - `-w` Weight between content and style terms fro the loss function
  - `-bn` Number of conv layer blocks in the architecture

To visualize your training:
```bash
tensorboard --logdir=models/GrainNet
```

### Training the grain size estimator

```bash
python train_cls.py -i './input/path'
```
Required:
  - `-i` Path to the folder containing the **grainy** images

Options:
  - `-s` Folder in which to save the model and its checkpoint
  - `-lr` Learning rate
  - `-bs` Batch size
  - `-ne` Number of epochs
  - `-ss` Step size for optimizer scheduler
  - `-ga` Gamma for optimizer scheduler

To visualize your training:
```bash
tensorboard --logdir=models/Classifier
```

## Running a pretrained model

```bash
python edit.py -i ./image.jpg
```
Options:
  - `-i` Path to the input image.
  - `-o` Path to the output image. default: 'output.png'
  - `-gs` Grain size used, it can be any value between 0.01 and 0.8. default: 0.1
  - `-s` Seed. default: Random based on datetime
  - `-m` Path to the pretrained network file. default: './models/GrainNet/default/grainnet.pt'

## Licence

All rights reserved. The code is released for academic research use only.

## Citation

If you use our code/data, please cite our paper.

## Acknowledgments

This work uses existing libraries for its evaluation.
We thank [lpips](https://pypi.org/project/lpips/), [DISTS_pytorch](https://pypi.org/project/DISTS-pytorch/) and the following repository for its [SIFID](https://github.com/tamarott/SinGAN/blob/master/SIFID/sifid_score.py) implementation.
