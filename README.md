# Neural Film Grain Rendering

Official implementation of the paper [*Neural Film Grain Rendering*](https://hal.science/hal-04667141)

![Python 3.9](https://img.shields.io/badge/Python-3.9-yellow.svg)
![pytorch 1.12.0](https://img.shields.io/badge/Pytorch-1.12.0-blue.svg)

![image](./images/teaser.png)
**Figure:** *Film Grain rendered by our method*

> **Neural Film Grain Rendering** <br>
>  Gwilherm Lesné, Yann Gousseau, Saïd Ladjal, Alasdair Newson <br>
>  LTCI, Télécom Paris 
>  ISIR, Sorbonne Université<br>

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
**N.B.** This code relies on the official CUDA implementation of [A Stochastic Film Grain Model for Resolution‐Independent Rendering]{https://onlinelibrary.wiley.com/doi/10.1111/cgf.13159} for creating the database and computing the metrics. Please follow the **requirements** [here](https://github.com/alasdairnewson/film_grain_rendering_gpu) to use it.

## Getting data

You have two options: 
- You want to use the data we used :
  - Download the files [here](XXX)
  - Unzip `XXX.zip` in `data` folder
- You want to train the model on your own data:
  - Get the code [here](https://github.com/alasdairnewson/film_grain_rendering_gpu) and follow the steps to be able to run it properly
  - Add the path to the executable (film_grain_rendering_main) in `dataset.py`
  - Create the dataset:
    ```
    python dataset.py ./data/path_to__clean_dataset -m ./data/path_to_newson_method/
    ```

## Training

```bash
python train.py
```
Options:
  - `-lr` Learning rate
  - `-ln` Number of dense layers to use for both your encoder and decoder.
  - `-bs` Batch size
  - `-ne` Number of epochs
  - `-bn` Tells if you put batch normalisation layers in your auto-encoder
  - `-aw` Weight for attribute loss term
  - `-dw` Weight for disentanglement loss term
  - `-k` Number of PCA dimensions to keep
  - `-dl` Disentanglement loss, you may want to disentangle the latent space or take into account the natural correlations of your data. The default is the former. For the latter, the CelebA attribute correlations will be used.
  - `-ai` Indices of the attributes you want to take into account
  - `-df` Path to the folder where the data is stored (`data.pkl` and `label.pkl` files)
  - `-sp` Path to the folder you want to save the model in

To visualize your training:
```bash
tensorboard --logdir=models
```

## Running a pretrained model

```bash
python edit.py -i ./input/path -o ./output/path -gs grain_size
```
Options:
  - `-i` Path to the input image.
  - `-o` Path to the output image.
  - `-s` Grain size used, it can be any value between 0.01 and 0.8.
  - `-s` Seed
  - `-m` Path to the pretrained network file, default: './models/GrainNet/grainnet.pt'

## Licence

All rights reserved. The code is released for academic research use only.

## Citation

If you use our code/data, please cite our paper.

## Acknowledgments

This work uses existing libraries for its evaluation.
We thank [lpips]{https://pypi.org/project/lpips/}, [DISTS_pytorch]{https://pypi.org/project/DISTS-pytorch/} and the following repository for its [SIFID]{https://github.com/tamarott/SinGAN/blob/master/SIFID/sifid_score.py} implementation.
