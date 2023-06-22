# brain2face

- [DALLE-2 paper](https://arxiv.org/pdf/2204.06125.pdf)

## UHD

### Overview

<div align="center"><img src="assets/overview.jpeg" width=700></div>

### TODOs

- [ ] Accept negative shifts (sessions where video recording started before EEG recording)
- [ ] Preprocessingの出力を.npyから.h5にする

### Usage

- Run preprocess

```bash
nohup python brain2face/preprocs/uhd.py start_subj=0 end_subj=8 > logs/uhd/out1.log &
nohup python brain2face/preprocs/uhd.py start_subj=8 end_subj=16 > logs/uhd/out2.log &
nohup python brain2face/preprocs/uhd.py start_subj=16 end_subj=22 > logs/uhd/out3.log &
```

- Run CLIP training

```bash
# Specify sweep configuration from .yaml
nohup python brain2face/train_clip.py config_path=uhd/image.yaml sweep=True > logs/uhd/sweep_clip.log &
```

<br>

## Hayashi Lab @ AIST

### Usage

- Submodule [encoder4editing](https://github.com/SeanNobel/encoder4editing)

- Download StyleGAN inversion model trained on FFHQ StyleGAN
```bash
cd encoder4editing/weights
gdown https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT
```

- Run preprocess (using 4 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0 nohup python brain2face/preprocs/stylegan.py start_subj=0 end_subj=8 > logs/ica/out1.log &
CUDA_VISIBLE_DEVICES=1 nohup python brain2face/preprocs/stylegan.py start_subj=8 end_subj=16 > logs/ica/out2.log &
CUDA_VISIBLE_DEVICES=2 nohup python brain2face/preprocs/stylegan.py start_subj=16 end_subj=24 > logs/ica/out3.log &
CUDA_VISIBLE_DEVICES=3 nohup python brain2face/preprocs/stylegan.py start_subj=24 end_subj=32 > logs/ica/out4.log &
```

<br>

## Yanagisawa Lab @ Osaka Univ.