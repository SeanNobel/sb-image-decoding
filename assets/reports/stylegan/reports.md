# データセット

- アラヤでFace2Brain用に取得したデータセット．脳波被験者はUnityの運転ゲーム課題を行なっている．

- 被験者5名，1人あたり6セッション程度で計32セッション．1セッション1時間程度で，総計すると30時間程度のEEGデータ．


## 顔動画の前処理

1. 30Hzの動画全フレームをStyleGAN latentに変換 `( time, styles=18, features=512 )`

1. 実時間で3秒分を1サンプルとして対照学習をするので，それ用に変形 `( samples, time=90, 18, 512 )`

1. とりあえず5-8層を取ってくる `( samples, 90, 4, 512 )`

1. 顔動画が30Hz，EEGが120Hzでちょうど4層とると形が合うので，とりあえずそこをconcatしてしまって学習を試す `( samples, 360, 512 )`

1. EEGに合わせてトランスポーズ `( samples, 512, 360 )`


## EEGの前処理と，BrainEncoderによるCLIP空間へのマッピング

- 120Hzにダウンサンプリング，ベースライン・スケーリング・クリッピングなどの前処理 `( samples, channels=32, time=360 )`

- BrainEncoderでエンコード（時間方向には長さを変えない） `( samples, 512, 360 )`


# CLIP

- バッチ内で`( 512, 360 )`の行列の類似度行列をとる


## テスト

- バッチサイズ==64でzero-shot classificationのtop1とtop10．なのでチャンスレベルそれぞれ一応1.6%, 16%くらいだが，注意点としてはdeep split, shallow splitともに被験者内でスプリットしているので，訓練・テスト間の相関が大きく，実際のチャンスレベルはもう少し高いと思われる．

  - Shallow split: 1時間のセッションを3秒にセグメントしてから，ランダムにtrain/test split

  - Deep split: 1時間のセッションを3秒にセグメントしてから，最初の80%（50分くらい）をtrainへ，最後の20%をテストへ

  - Subject random split (これからcross validationで実行予定): 5人の被験者のうち1人をテスト用被験者とする

<figure align="center"><img src="assets/reports/230512.png" width=100% alt="230512"></figure>