# neuro-diffusion

## Overview

<div align="center"><img src="assets/neuro-diffusion.jpeg" width=700></div>

References

- Hierarchical Text-Conditional Image Generation with CLIP Latents ([Ramesh et al., Apr 2022](https://arxiv.org/pdf/2204.06125.pdf))

### DALLE-2 Video

References

- Video Diffusion Models ([Ho et al., Apr 2022](https://arxiv.org/abs/2204.03458))

- Imagen Video: High Definition Video Generation with Diffusion Models ([Ho et al., Oct 2022](https://arxiv.org/abs/2210.02303))

```bash
cd dalle2_video/
pip install -e .
```

## Status

### YLab GOD

- (9/19) 100エポックでもGTに近い画像は生成されなかった．
  - Decoderだけの評価としてimage_emedsから直接生成してみたところ，少しGTに似た画像が生成された．
  - Decoderを800エポックで再訓練．

- (9/8) 訓練データに対して，パイプラインに通して画像を生成してみた（`generated/ylabgod`）．Decoderを20エポックしか訓練していなかったので，100エポックにして再訓練．

- (9/7) [Priorの訓練200エポックが完了](https://wandb.ai/sensho/nd_god_prior/runs/u70r7dga?workspace=user-sensho)．Image decoderの訓練を開始．

- (9/6) Priorの訓練が50エポックでは足りなかった（+ wandbを使えていなかった）ので200エポックにして回し直す．

- (8/25) 画像掲示がそもそも500msなので前回のsweepに意味がないことに気づいた．test accは上がっていないが，とりあえずパイプライン全体を訓練して訓練データで生成画像をGTと比較してみる．現在priorを訓練中．

- (8/14) 画像掲示からseq_len=0.5秒間のECoGを使っていたので情報が乗っていないのかもしれない．画像掲示から0.5-1.0, 1.0-1.5, 1.5-2.0秒間のECoGで訓練sweep
  - (9/19) ↑画像掲示が0.5秒なので意味がないはずだが，やらなかった？（画像掲示間のインターバルはゼロ秒！）

- (8/10) mixed_deepでもまったくtest accが上がらなかったので，mixed_shallowを作成，CLIP sweepを開始．ついでにconv blockのkernel sizeを3, 5, 7でsweep

- (8/7) 訓練とテストを混ぜてからdeep splitするmixed_deep splitを作成，CLIP sweepを開始．そのパフォーマンスによってはpriorの訓練に移る．

### UHD

- (10/23) Unet3DEncoderのd_drop=0.1, 0.2は0.3よりもtest accが下がった．d_drop=0.4, 0.5を試す．

- (10/17)
  - [Apollo-5] `Unet3DEncoder(d_drop=0.3)`, `ViViTReduceTime`ともに学習ストップ．`Unet3DEncoder(d_drop=0.3)`で推論し，prior/decoder用のデータセットを作成，ms4-5に移動．さらに，Unet3DEncoderのd_drop=0.1, 0.2を別プロセスで開始．
  - [ms4-5] `Unet3DEncoder(d_drop=0.3)`で作ったlatentsでpriorを訓練，終了．さらに，decoderの訓練を開始．

- (10/10) 現在decoderの学習に使用しているCLIP embeddingsがどの学習済みモデルからのものなのか定かでなくなった＆ちゃんとサーチで性能比較をしていなかったので，CLIPの学習をやりなおす．Unet3DEncoderとViViTReduceTimeともにd_drop={0.3, 0.2, 0.4}でsweepを開始．
  - Preprocessedデータのface.h5をchunkingした．

- (10/6) EMAが原因ではない．DeepSpeedを使わず1GPUなら両方のUNetの学習が進んだ．DeepSpeedのときはunetsを直接継承していないdecoderしかaccelerator.prepareに渡していないことが原因かもしれないので，次はそれをやる．

- (10/5) ms4-5に移行して昨日フルサイズモデルを訓練し始めたが，unet1の学習が進まない．EMAを使用していることが原因かもしれないので，EMAをオフにして新しい訓練を開始．

- (8/21) Video Decoderは訓練途中のものではあるが，パイプライン全体を走らせて初の3D U-Netでの動画生成をした．結果，ちゃんと動画が生成された．Video Decoderがbatch size=1でも訓練できないためモデルを小さくしている（それでも訓練に非常に時間がかかる）ので，Lightning FSDPかAccelerate DeepSpeedを導入してmodel parallelismを試す．

- (8/18) Unet3DEncoderを使ったCLIP学習はViViTほどパフォーマンスが出ない．ViViTReduceTimeを実装，CLIP学習を開始．学習途中のUnet3DEncoderではあるが，static embeddingでパイプラインを通してデバッグするための`eval_clip.py`を開始．

- (8/17) CLIP学習用のUnet3dEncoderを実装．

- (8/16) Unet3Dを実装．

- (8/15) Video Decoderを30Hzのまま訓練することはGPUメモリの不足でできないので，16サンプル(5.3Hzくらい)でやりなおし．パイプラインを走らせるときもDatasetやCollateFnでリサンプリングする．

- (8/10) CLIP学習がおわっていないがevalを回してclip_embdsをいったん作ってprior訓練してみる．

- (8/9) dalle2_videoのサンプリングに関するメソッドが動画に対応したので，パイプライン走らせられる状態になった．依然時間次元30HzのCLIP学習待ち．
  - 学習済みCLIPモデル：`runs/uhd/video/d_drop-0.3_/`


- (8/8) Diffusion priorの訓練でロスがnanになったが，元々はならず大きな変更も加えていない．パイプライン全体の訓練を最初からやってみる．今は30Hzのまま時間次元ありCLIPを学習中．

- (8/8) Downsamplingスクリプトがおわったが，CLIP embeddingが時間次元を持ったままのパイプラインを作ることにした．時間次元をバッチ次元とflattenしてのprior trainingを走らせている．Video decoderの訓練はすでに前やってある（分散を学習しないことで学習を安定させたやつ）．

- (8/7) Videoを一つのCLIP embeddingにしてやるパイプラインのためのdownsamplingスクリプトのおわり待ち．おわったら実行：`python brain2face/train_clip.py config_path=uhd/video/clip.yaml`

## TODOs

Aug

- [x] dalle2_videoのサンプリングメソッドたちを動画に対応させる．

Jul

- [x] 他のconfigファイルの内容をimage.yamlに合わせる
- [x] Preprocessingのfaceの出力を.npyから.h5にする
- [x] args.face.encoded=Trueでも元の画像を保存できるようにする
  - [x] この過程でargs.face.pretrainedに変わったので，YLabGOD以外も対応させる
  - [x] train_clip.pyも今動かない状態
- [ ] Uknown subjectのとき全subject layersの出力の平均を取るようにしているが，これで良いのか考える
  - Known subjectでのactivationとの類似度とかを取ってそれで重みづけするとか
- [ ] 毎回のsweepで一つchance modelが走るようにする
- [x] YLabGOD以外も`y_reformer`を`loader`にする
- YLab
  - [x] チャネル空間座標の導入
  - [x] 実時間を3秒からハイパラにする
    - [x] Segmentingしないpreprocessingの追加
    - [x] Datasetsに内部でsegmentingするモードを追加
  - [ ] priorを訓練して本物のimage imbとの相関を取る
  - [x] AU_rだけを使う
- UHD
  - [ ] 負のシフトを受け付ける（録画がEEG記録の前に始まってしまった？セッション）
- Decoder training
  - [x] `NeuroDiffusionCLIPEmbVideoDataset`のメモリー問題を解決

<br>

## Preprocessing

### Yanagisawa Lab GOD

```bash
python brain2face/preprocs/ylab_god.py
```

### Yanagisawa Lab OpenFace (subject E0030)

```bash
python brain2face/preprocs/ylab_e0030.py
```

### UHD

```bash
nohup python brain2face/preprocs/uhd.py start_subj=0 end_subj=8 > logs/uhd/out1.log &
nohup python brain2face/preprocs/uhd.py start_subj=8 end_subj=16 > logs/uhd/out2.log &
nohup python brain2face/preprocs/uhd.py start_subj=16 end_subj=22 > logs/uhd/out3.log &
```

### Hayashi Lab @ AIST

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

## CLIP training

```bash
# Normal
python brain2face/train_clip.py config_path={path to config}.yaml 

# Sweep
nohup python brain2face/train_clip.py config_path={path to config}.yaml sweep=True > logs/{path to log}.log &
```

<br>

## CLIP evaluation (save CLIP embeds + corresponding images / videos)

```bash
python brain2face/eval_clip.py config_path={path to config}.yaml
```

<br>

## DALLE-2 prior training

```bash
python brain2face/train_prior.py config_path={path to config}.yaml
```

<br>

## DALLE-2 decoder training

### Image

```bash
nohup python brain2face/train_image_decoder.py config_path={path to config}.yaml > logs/{path to log}.log &
```

### Video

```bash
nohup python brain2face/train_video_decoder.py config_path={path to config}.yaml > logs/{path to log}.log &
```

### Video (with DeepSpeed)

- First run `accelerate config` with answering 'no' to the question 'Do you want to specify a json file to a DeepSpeed config?'

- One cannot use nohup with many distributed training APIs (https://github.com/pytorch/pytorch/issues/67538). Use tmux instead.

```bash
accelerate launch brain2face/train_video_decoder.py config_path={path to config}.yaml
```

```bash
tmux new -s train-video-decoder

CUDA_VISIBLE_DEVICES=2,3 accelerate launch brain2face/train_video_decoder.py config_path=uhd/video/decoder/static-emb.yaml use_wandb=True

[Ctrl+b -> d] # detach from tmux session

tmux ls # check running sessions

tmux attach -t train-video-decoder

tmux kill-session -t train-video-decoder
```

```bash
# Edit accelerate config if necessary (e.g. main_process_port)
vi ~/.cache/huggingface/accelerate/default_config.yaml
```

## Finally, run the pipeline and generate images / videos from EEG

### Image

```bash
python brain2face/eval_image_pipeline.py config_path={path to config}.yaml
```

### Video

```bash
python brain2face/eval_video_pipeline.py config_path={path to config}.yaml
```

<br>

## Deprecated

### Distributed DALLE-2 prior training for image (deprecated)

```bash
python brain2face/distributed_train_prior.py
```

### Distributed DALLE-2 decoder training for image (deprecated)

```bash
# tar face_images
bash tar_face_images.sh
# login to huggingface hub
huggingface-cli login
# create a repository in huggingface whose name matches tracker.save.huggingface_repo in decoder.json
# need to create this directory manually
mkdir .tracker_data
# Run distributedtraining
python brain2face/train_decoder_distributed.py
```