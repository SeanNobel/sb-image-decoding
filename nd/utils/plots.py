import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional
from termcolor import cprint

matplotlib.use("Agg")


def plot_latents_2d(
    latents: np.ndarray,
    classes: np.ndarray,
    save_path: Optional[str] = None,
    cmap: str = "gist_rainbow",
) -> Optional[plt.Figure]:
    """
    latents ( samples, dim )
    classes ( samples, )
    """
    if latents.ndim > 2:
        cprint(f"Flattening latents with more than 2 dimensions ({latents.shape}) for plot.", "yellow")  # fmt: skip
        latents = latents.reshape(latents.shape[0], -1)

    classes = classes.astype(float) / classes.max()

    fig, axs = plt.subplots(ncols=5, figsize=(20, 5), tight_layout=True)
    fig.suptitle(f"Originally {latents.shape[1]} dimensions")

    latents_reduced = PCA(n_components=2).fit_transform(latents)
    axs[0].scatter(*latents_reduced.T, c=classes, cmap=cmap)
    axs[0].set_title("PCA")

    for perplexity, ax in zip([2, 10, 40, 100], axs[1:]):
        print(f"Running t-SNE with perplexity={perplexity}...")
        latents_reduced = TSNE(n_components=2, perplexity=perplexity).fit_transform(latents)  # fmt: skip

        ax.scatter(*latents_reduced.T, c=classes, cmap=cmap)
        ax.set_title(f"t-SNE (perplexity={perplexity})")

    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        fig.savefig(save_path)
    else:
        return fig
