import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional, List
from termcolor import cprint

matplotlib.use("Agg")


def plot_latents_2d(
    latents: np.ndarray,
    classes: np.ndarray,
    perplexities: List[int] = [2, 10, 40, 100],
    pca: bool = True,
    save_path: Optional[str] = None,
    cmap: str = "gist_rainbow",
    off_axis: bool = False,
) -> Optional[plt.Figure]:
    """
    latents ( samples, dim )
    classes ( samples, )
    """
    if latents.ndim > 2:
        cprint(f"Flattening latents with more than 2 dimensions ({latents.shape}) for plot.", "yellow")  # fmt: skip
        latents = latents.reshape(latents.shape[0], -1)

    classes = classes.astype(float) / classes.max()

    ncols = len(perplexities) + pca
    fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 4, 5), tight_layout=True, squeeze=False)
    # NOTE: removing row axis that's always 1
    axs = axs[0]

    if not off_axis:
        fig.suptitle(f"Originally {latents.shape[1]} dimensions")

    if pca:
        latents_reduced = PCA(n_components=2).fit_transform(latents)
        axs[0].scatter(*latents_reduced.T, c=classes, cmap=cmap)
        if off_axis:
            axs[0].axis("off")
        else:
            axs[0].set_title("PCA")

    axs_tsne = axs[1:] if pca else axs
    for perplexity, ax in zip(perplexities, axs_tsne):
        print(f"Running t-SNE with perplexity={perplexity}...")
        latents_reduced = TSNE(n_components=2, perplexity=perplexity).fit_transform(latents)  # fmt: skip

        ax.scatter(*latents_reduced.T, c=classes, cmap=cmap, s=2)
        if off_axis:
            ax.axis("off")
        else:
            ax.set_title(f"t-SNE (perplexity={perplexity})")

    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        fig.savefig(save_path)
    else:
        return fig


def plot_2d_latents_with_sorted_categories(
    train_Z, train_Y, train_categories, test_Z, test_Y, test_categories
):
    sorted_high_categories = ["bird", "insect", "plant", "body part", "animal", "vegetable", "fruit", "drink", "dessert", "food", "clothing", "clothing accessory", "furniture", "home decor", "toy", "kitchen appliance", "kitchen tool", "office supply", "sports equipment", "medical equipment", "tool", "musical instrument", "electronic device", "weapon", "vehicle", "part of car", "container"]  # fmt: skip

    hc_path = (
        "/mnt/tsukuyomi/things/osfstorage/THINGS/27 higher-level categories/category_mat_manual.tsv"
    )
    high_categories = np.loadtxt(hc_path, dtype=int, delimiter="\t", skiprows=1)
    # ( 1854, 27 )

    unsorted_high_categories = np.loadtxt(hc_path, dtype=str, delimiter="\t")[0]

    arange = np.arange(len(high_categories))  # ( 1854, )
    argsort = []
    for shc in sorted_high_categories:
        idxs_in_hc = np.where(high_categories[:, unsorted_high_categories == shc] == 1)[0]  # fmt: skip
        for i in idxs_in_hc:
            if arange[i] != -1:
                argsort.append(i)
                arange[i] = -1

    # Appending images that don't belong to any high category
    for i in arange:
        if i != -1:
            argsort.append(i)

    argsort = {str(j): i for i, j in enumerate(argsort)}
    argsort.update({"1854": 1854})

    def sort_categories(categories: np.ndarray) -> np.ndarray:
        return np.array([argsort[str(i)] for i in categories])

    fig, ax = plt.subplots(1, 4, figsize=(20, 5), tight_layout=True)
    ax[0].scatter(*train_Y.T, c=sort_categories(train_categories), s=1, cmap="gist_rainbow")
    ax[0].set_title("Image train")
    ax[1].scatter(*test_Y.T, c=sort_categories(test_categories), s=1, cmap="gist_rainbow")
    ax[1].set_title("Image test")
    ax[2].scatter(*train_Z.T, c=sort_categories(train_categories), s=1, cmap="gist_rainbow")
    ax[2].set_title("Text (MEG) train")
    ax[3].scatter(*test_Z.T, c=sort_categories(test_categories), s=1, cmap="gist_rainbow")
    ax[3].set_title("Text (MEG) test")

    fig.canvas.draw()
    plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return plot
