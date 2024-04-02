"""
This script only produces text csv files with different levels of noises added.
Need to manually copy other files such as Images_P1.pt and Captions.csv to the 
preprocessed data directory beforehand.
"""

import os, sys
import numpy as np
import csv
import hydra
from omegaconf import DictConfig
from typing import List
from termcolor import cprint

NOISE_LEVELS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def add_noise(
    texts: List[str], noise_level: float, clamp_length: int = 77
) -> List[str]:
    """Randomly replace characters in text with noise_level probability."""
    noisy_texts = []
    for i, text in enumerate(texts):
        noisy_text = ""

        for c, char in enumerate(text):
            if np.random.rand() < noise_level:
                noisy_text += chr(np.random.randint(97, 123))
            else:
                noisy_text += char

            if c == clamp_length - 1:
                break

        noisy_texts.append(noisy_text)

        # cprint(text, "cyan")
        # cprint(noisy_text, "yellow")
        # if i > 10:
        #     sys.exit()

    return noisy_texts


@hydra.main(
    version_base=None, config_path="../../configs/thingsmeg", config_name="clip_text"
)
def run(args: DictConfig) -> None:
    preproc_dir = os.path.join(args.preprocessed_data_dir, args.preproc_name)

    texts_path = os.path.join(preproc_dir, "Captions.csv")
    keys = [key for key, _ in csv.reader(open(texts_path, "r"))]
    orig_texts = [text for _, text in csv.reader(open(texts_path, "r"))]

    for noise_level in NOISE_LEVELS:
        save_dir = os.path.join(preproc_dir, f"noise_level_{noise_level}")
        os.makedirs(save_dir, exist_ok=True)

        for n in range(args.num_noises):
            texts = add_noise(orig_texts, noise_level, clamp_length=77)

            with open(os.path.join(save_dir, f"Texts_N{n}.csv"), "w") as f:
                writer = csv.writer(f)
                writer.writerows([key, text] for key, text in zip(keys, texts))


if __name__ == "__main__":
    run()
