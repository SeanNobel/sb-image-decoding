# brain2face

## encoder4editing

- Submodule -> https://github.com/SeanNobel/encoder4editing

```bash
cd encoder4editing/weights # weights folder was created
gdown https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT # FFHQ Inversion model
```

## CLIP

- Submodule -> https://github.com/SeanNobel/speech-decoding/tree/brennan-fixes

```bash
pip install -r speech-decoding/requirements.txt
```

## Dataset

- Submodule -> https://github.com/arayabrain/f2b-contrastive

## Install

```bash
pip install -r requirements.txt
```

## Run preprocess

```bash
CUDA_VISIBLE_DEVICES=0 nohup python preproc.py start_subj=0 end_subj=8 > out1.log &
CUDA_VISIBLE_DEVICES=1 nohup python preproc.py start_subj=8 end_subj=16 > out2.log &
CUDA_VISIBLE_DEVICES=2 nohup python preproc.py start_subj=16 end_subj=24 > out3.log &
CUDA_VISIBLE_DEVICES=3 nohup python preproc.py start_subj=24 end_subj=32 > out4.log &
```