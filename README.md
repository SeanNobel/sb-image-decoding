# brain2face

## StyleGAN

[CelebA Dataset](https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P)

```bash
pip install -r annotated_deep_learning_paper_implementations/requirements.txt
pip install -e annotated_deep_learning_paper_implementations/

cd annotated_deep_learning_paper_implementations/data/stylegan2/
gdown https://drive.google.com/uc?id=1O89DVCoWsMhrIF3G8-wMOJ0h7LukmMdP # 256 x 256
unzip data256x256.zip

cd ../../..
python annotated_deep_learning_paper_implementations/labml_nn/gan/stylegan/experiment.py
```

```mermaid
flowchart TD

```