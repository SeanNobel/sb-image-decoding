import sys
import numpy as np
import torch
from glob import glob
from natsort import natsorted
from PIL import Image
import clip
import matplotlib.pyplot as plt
from tqdm import tqdm


def run():
    device = "cuda:0"
    dir = "data/preprocessed/imageneteeg/0_init/"
    fig_dir = "figures/sb_samples/ae_100000_uncond/"
    label_names = np.loadtxt("data/raw/eeg_cvpr_2017/classes.txt", dtype=str)

    test_idxs = torch.load(dir + "test_idxs/cv1.pt")

    labels = torch.load(dir + "labels.pt").take(test_idxs)

    clip_model, preprocess = clip.load("ViT-L/14")
    clip_model = clip_model.eval().requires_grad_(False).to(device)

    image_paths = natsorted(glob(fig_dir + "with_gt/*.png"))
    clip_gt, clip_gen = [], []
    for path in tqdm(image_paths):
        image = np.array(Image.open(path).convert("RGB"))
        image_gt, image_gen = np.split(image, 2, axis=0)
        assert image_gt.shape == image_gen.shape and image_gt.shape[0] == image_gt.shape[1]

        image_gt, image_gen = Image.fromarray(image_gt), Image.fromarray(image_gen)

        clip_gt_ = clip_model.encode_image(preprocess(image_gt).unsqueeze(0).to(device)).float()
        clip_gen_ = clip_model.encode_image(preprocess(image_gen).unsqueeze(0).to(device)).float()

        clip_gt.append(clip_gt_)
        clip_gen.append(clip_gen_)

    clip_gt = torch.cat(clip_gt).cpu()  # ( 1997, 768 )
    clip_gen = torch.cat(clip_gen).cpu()  # ( 1997, 768 )

    clip_gt /= clip_gt.norm(dim=-1, keepdim=True)
    clip_gen /= clip_gen.norm(dim=-1, keepdim=True)

    sim = (clip_gt @ clip_gen.T).numpy()  # ( 1997, 1997 )

    # Average over labels
    sim_label = np.zeros((labels.max() + 1, labels.max() + 1))
    counts = np.zeros_like(sim_label)
    for i in tqdm(range(sim.shape[0])):
        for j in range(sim.shape[1]):
            sim_label[labels[i], labels[j]] += sim[i, j]
            counts[labels[i], labels[j]] += 1

    sim_label /= counts

    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
    im = ax.imshow(sim_label, cmap="viridis")
    ax.set_yticks(range(len(label_names)))
    ax.set_yticklabels(label_names, fontdict={"fontsize": 13})
    ax.tick_params(left=False, bottom=False, labelbottom=False)
    plt.colorbar(im, fraction=0.047, pad=0.01)
    fig.savefig(fig_dir + "grid/confusion.png")


if __name__ == "__main__":
    run()
