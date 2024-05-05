import os, sys
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
import mne
from sklearn.preprocessing import RobustScaler
from PIL import Image
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from termcolor import cprint
from itertools import product

import clip
from uvit import libs

from nd.utils.brain_preproc import baseline_correction


def make_split(subject_idxs: torch.Tensor, labels: torch.Tensor, train_ratio: float = 0.8):
    """Makes splits so that train/test contains all subjects and classes.
    There are 6 subjects and 40 classes (50 images per class).
    This is not used.
    Args:
        subject_idxs (torch.Tensor): _description_
        labels (torch.Tensor): _description_
    """
    train_idxs = []
    test_idxs = []
    _total_len = []
    for i, (subject_idx, label) in enumerate(
        product(set(subject_idxs.tolist()), set(labels.tolist()))
    ):
        idxs = torch.where(torch.logical_and(subject_idxs == subject_idx, labels == label))[0]
        _total_len.append(len(idxs))
        if len(idxs) != 50:
            cprint(f"Subject={subject_idx} label={label} has {len(idxs)} images", "yellow")

        train_samples = int(len(idxs) * train_ratio)
        train_idxs.extend(idxs[:train_samples])
        test_idxs.extend(idxs[train_samples:])

    assert i == 6 * 40 - 1
    assert np.sum(_total_len) == len(subject_idxs)
    assert len(train_idxs) + len(test_idxs) == len(subject_idxs)

    return torch.tensor(train_idxs).sort().values, torch.tensor(test_idxs).sort().values


USE_PROCESSED = False


@hydra.main(version_base=None, config_path="../../configs/imageneteeg", config_name="clip")
def run(args: DictConfig) -> None:
    save_dir = os.path.join(args.preproc_dir, args.preproc_name)
    images_dir = os.path.join(save_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "train_idxs"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "test_idxs"), exist_ok=True)

    if USE_PROCESSED:
        data = torch.load(os.path.join(args.eeg_dir, "eeg_5_95_std.pth"))
    else:
        data = torch.load(os.path.join(args.eeg_dir, "eeg_signals_raw_with_mean_std.pth"))

    # make it starts from 0
    subject_idxs = torch.tensor([d["subject"] for d in data["dataset"]]) - 1
    torch.save(subject_idxs, os.path.join(save_dir, "subject_idxs.pt"))

    labels = torch.tensor([d["label"] for d in data["dataset"]])
    torch.save(labels, os.path.join(save_dir, "labels.pt"))

    splits = torch.load(os.path.join(args.eeg_dir, "block_splits_by_image_all.pth"))
    # train_idxs, test_idxs = make_split(subject_idxs, labels)
    for cv in range(len(splits["splits"])):
        train_idxs = splits["splits"][cv]["train"] + splits["splits"][cv]["val"]
        train_idxs = torch.tensor(train_idxs).sort().values

        test_idxs = torch.tensor(splits["splits"][cv]["test"])

        torch.save(train_idxs, os.path.join(save_dir, "train_idxs", f"cv{cv+1}.pt"))
        torch.save(test_idxs, os.path.join(save_dir, "test_idxs", f"cv{cv+1}.pt"))

    cprint(f"Saved subject_idxs {subject_idxs.shape}, {subject_idxs.dtype} | labels {labels.shape}, {labels.dtype} | train_idxs {train_idxs.shape}, {train_idxs.dtype} | test_idxs {test_idxs.shape}, {test_idxs.dtype} to {save_dir}", "cyan")  # fmt: skip

    # -----------------
    #        EEG
    # -----------------
    if not args.skip_eeg:
        # The dataset authors say 'we discarded the first 20 samples (20 ms) to reduce interference from the previous image and then cut the signal to a common length of 440 samples'. https://github.com/perceivelab/eeg_visual_classification
        eeg = torch.stack([d["eeg"][:, 20:][:, :440] for d in data["dataset"]])
        eeg = eeg.numpy().astype(np.float64)

        if args.resample_freq != 1000:
            print(f"Resampling EEG to {args.resample_freq}.")
            eeg = mne.filter.resample(eeg, down=1000 / args.resample_freq)  # ( segments, c, t )

        if not USE_PROCESSED:
            print("Applying notch filter.")
            eeg = mne.filter.notch_filter(eeg, Fs=args.resample_freq, freqs=[50, 100, 150])

            # Channel-wise scaling
            print(f"Scaling and clamping with +/- {args.clamp_lim}.")
            eeg = RobustScaler().fit_transform(eeg.reshape(-1, eeg.shape[-1])).reshape(eeg.shape)
            eeg = eeg.clip(min=-args.clamp_lim, max=args.clamp_lim)

            eeg = baseline_correction(eeg, baseline_len_samp=int(eeg.shape[-1] * 0.2))

        eeg = torch.from_numpy(eeg).to(torch.float32)
        torch.save(eeg, os.path.join(save_dir, "eeg.pt"))
        cprint(f"Saved EEG {eeg.shape}, {eeg.dtype} to {save_dir}", "cyan")

    # -----------------
    #      Images
    # -----------------
    if not args.skip_images:
        device = "cuda:0"

        autoencoder = libs.autoencoder.get_model("uvit/assets/stable-diffusion/autoencoder_kl.pth")
        # , scale_factor=0.23010
        autoencoder.to(device)

        clip_model, preprocess = clip.load(args.vision.pretrained_model)
        clip_model = clip_model.eval().requires_grad_(False).to(device)

        image_set_paths = [
            os.path.join(args.images_dir, name.split("_")[0], f"{name}.JPEG")
            for name in data["images"]
        ]
        clip_embeds, moments = [], []
        for path in tqdm(image_set_paths, desc="Embedding images"):
            image = Image.open(path).convert("RGB")

            # ------------------------------
            #           CLIP embeds
            # ------------------------------
            clip_embed = clip_model.encode_image(preprocess(image).unsqueeze(0).to(device)).float()
            clip_embeds.append(clip_embed)

            # ------------------------------
            #    Stable Diffusion moments
            # ------------------------------
            crop_size = min(image.size)
            image = TF.center_crop(image, [crop_size, crop_size])
            image = TF.resize(image, 256, Image.LANCZOS)
            image.save(os.path.join(images_dir, os.path.basename(path).replace(".JPEG", ".jpg")))

            image = np.array(image, dtype=np.float32) / 127.5 - 1.0
            image = torch.from_numpy(image).permute(2, 0, 1)

            moment = autoencoder.encode_moments(image.unsqueeze(0).to(device))
            moments.append(moment)

        clip_embeds = torch.cat(clip_embeds).cpu()
        moments = torch.cat(moments).cpu()

        image_idxs = torch.tensor([d["image"] for d in data["dataset"]])
        clip_embeds = torch.index_select(clip_embeds, 0, image_idxs)
        moments = torch.index_select(moments, 0, image_idxs)

        torch.save(clip_embeds, os.path.join(save_dir, "images_clip.pt"))
        torch.save(moments, os.path.join(save_dir, "image_moments.pt"))
        cprint(f"Saved image CLIP-Vision embeds {clip_embeds.shape}, {clip_embeds.dtype} | Stable Diffusion moments {moments.shape}, {moments.dtype}", "cyan")  # fmt: skip


if __name__ == "__main__":
    run()
