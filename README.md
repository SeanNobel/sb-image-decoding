# brain2face

## UHD

### Overview

<div align="center"><img src="assets/overview.001.jpeg" width=700></div>

<br>

## Hayashi Lab @ AIST

### Install encoder4editing

- Submodule -> https://github.com/SeanNobel/encoder4editing

```bash
cd encoder4editing/weights # weights folder was created
gdown https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT # FFHQ Inversion model
```

### Run preprocess

```bash
CUDA_VISIBLE_DEVICES=0 nohup python brain2face/preprocs/stylegan.py start_subj=0 end_subj=8 > logs/ica/out1.log &
CUDA_VISIBLE_DEVICES=1 nohup python brain2face/preprocs/stylegan.py start_subj=8 end_subj=16 > logs/ica/out2.log &
CUDA_VISIBLE_DEVICES=2 nohup python brain2face/preprocs/stylegan.py start_subj=16 end_subj=24 > logs/ica/out3.log &
CUDA_VISIBLE_DEVICES=3 nohup python brain2face/preprocs/stylegan.py start_subj=24 end_subj=32 > logs/ica/out4.log &
```

<br>

## Yanagisawa Lab @ Osaka Univ.