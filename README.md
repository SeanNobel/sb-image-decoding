# Direct Generation of Images from EEG using Schrödinger Bridge

## Installation

Clone this repository with submodules.

Next, create a conda environment and install the dependencies and source code.

```bash
conda create -n sbid python=3.9
conda activate sbid
pip install pip==23.1.2
pip install -r requirements.txt
pip install -e .
```

## Data

Download and place ImageNetEEG dataset from [here](https://studentiunict-my.sharepoint.com/personal/concetto_spampinato_unict_it/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fconcetto%5Fspampinato%5Funict%5Fit%2FDocuments%2Fsito%5FPeRCeiVe%2Fdatasets%2Feeg%5Fcvpr%5F2017%2Fdata&viewid=0bec2d13%2D7140%2D4709%2D9a00%2D600d9e2f01e3&ga=1) and place all the files under `data/raw/eeg_cvpr_2017/` (or anywhere you like).

Download the ImageNet images used in the ImageNetEEG dataset and place them.

Change `root` and `images_dir` parameters in `configs/imageneteeg/clip.yaml` as we use that config file for preprocessing.

Run preprocessing.

```bash
python sbid/preprocs/imagenet_eeg.py
```

## Pre-training

### CLIP

```bash
python sbid/train_clip.py config_path=imageneteeg/clip.yaml
```

### Autoencoder

```bash
python sbid/train_autoencoder.py config_path=imageneteeg/autoencoder.yaml
```

## Schrödinger Bridge training

```bash
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 sbid/train_schrodinger_bridge.py --config=configs/imageneteeg/schrodinger_bridge_clip.py --config.train.name=clip
```

## Diffusion training

```bash
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 sbid/train_diffusion_cond.py --config=configs/imageneteeg/diffusion_cond_clip.py --config.train.name=clip
```

## Plotting

They are in `sbid/scripts/` or `notebooks/`.