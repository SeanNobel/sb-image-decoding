import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_latents_2d(
    latents: np.ndarray,
    classes: np.ndarray,
    epoch: int,
    save_dir: str,
    cmap: str = "gist_rainbow",
) -> None:
    """
    latents ( samples, dim )
    classes ( samples, )
    """
    classes = classes.astype(float) / classes.max()

    fig, axs = plt.subplots(ncols=5, figsize=(20, 5), tight_layout=True)
    fig.suptitle(f"Originally {latents.shape[1]} dimensions")

    latents_reduced = PCA(n_components=2).fit_transform(latents)
    axs[0].scatter(*latents_reduced.T, c=classes, cmap=cmap)
    axs[0].set_title("PCA")

    for perplexity, ax in zip([2, 10, 40, 100], axs[1:]):
        latents_reduced = TSNE(n_components=2, perplexity=perplexity).fit_transform(latents)  # fmt: skip

        ax.scatter(*latents_reduced.T, c=classes, cmap=cmap)
        ax.set_title(f"t-SNE (perplexity={perplexity})")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig.savefig(os.path.join(save_dir, f"epoch{epoch}.png"))
