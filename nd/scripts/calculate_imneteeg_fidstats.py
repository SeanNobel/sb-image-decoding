import os, sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
from tqdm import tqdm
from glob import glob

from pytorch_fid.fid_score import calculate_activation_statistics
from pytorch_fid.inception import InceptionV3


# fmt: off
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=50, help='Batch size to use')
parser.add_argument('--num-workers', type=int, help='Number of processes to use for data loading. Defaults to `min(8, num_cpus)`')
parser.add_argument('--device', type=str, default=None, help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--dims', type=int, default=2048, choices=list(InceptionV3.BLOCK_INDEX_BY_DIM), help='Dimensionality of Inception features to use. By default, uses pool3 features')
# parser.add_argument('--save-stats', action='store_true', help='Generate an npz archive from a directory of samples. The first path is used as input and the second as output.')
# parser.add_argument('path', type=str, nargs=2, help='Paths to the generated images or to .npz statistic files')
parser.add_argument('--save-path', type=str, default='fid_stats.npz', help='Path to save the statistics to')
parser.add_argument('--resize-to', type=int, default=None, help='Resize images to this size before calculating statistics')

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}
# fmt: on


def save_fid_stats(files, save_path, batch_size, device, dims, num_workers=1, resize_to=None):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    mu, sigma = calculate_activation_statistics(
        files, model, batch_size, dims, device, num_workers, resize_to
    )

    np.savez_compressed(save_path, mu=mu, sigma=sigma)


def main():
    filenames = [
        os.path.abspath(path) for path in glob("data/preprocessed/imageneteeg/0_init/images/*.jpg")
    ]

    # Args for FID calculation
    args = parser.parse_args()

    device = "cuda"

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    save_fid_stats(
        filenames,
        args.save_path,
        args.batch_size,
        device,
        args.dims,
        num_workers,
        args.resize_to,
    )


if __name__ == "__main__":
    main()
