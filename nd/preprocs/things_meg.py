"""
Here I use the preprocessed data in Hebart et al., 2023. It looks that scaling and baseline correction
is performed but clamping is not, which is different from the Meta paper. I resample the data from 200Hz
to 120Hz as in the Meta paper.
"""

import os, sys
import numpy as np
import mne
import torch
from PIL import Image
import csv
from sklearn.utils import gen_batches
from functools import partial
from termcolor import cprint
from natsort import natsorted
from tqdm import tqdm
from typing import List, Dict
import hydra
from omegaconf import DictConfig

from transformers import (
    AutoProcessor,
    CLIPVisionModel,
    BlipProcessor,
    BlipForConditionalGeneration,
)
import clip

from uvit.libs import clip as uvit_clip

from nd.utils.brain_preproc import scale_clamp
from nd.utils.train_utils import sequential_apply


@torch.no_grad()
def caption_images(y_list: List[str], processor, model, device) -> Dict[str, str]:
    """
    Generates captions for images using Salesforce/blip-image-captioning-large.
    Returns a dict of captions sorted by image names.
    """
    y_list = natsorted(y_list)

    num_samples = len(y_list)
    batch_size = 128
    pbar = tqdm(total=num_samples, desc="Generating captions (this is run only once)")
    slice = gen_batches(num_samples, batch_size)

    texts_dict = {}
    for _slice in slice:
        y_sublist = y_list[_slice]
        images = [Image.open(y).convert("RGB") for y in y_sublist]

        inputs = processor(images, return_tensors="pt").to(device)
        outputs = model.generate(**inputs).cpu()
        texts = [processor.decode(out, skip_special_tokens=True) for out in outputs]
        texts_dict.update(
            {os.path.splitext(os.path.basename(y))[0]: t for y, t in zip(y_sublist, texts)}  # fmt: skip
        )

        pbar.update(len(y_sublist))

    return texts_dict


@torch.no_grad()
def encode_images(y_list: List[str], preprocess, clip_model, device) -> torch.Tensor:
    """Encodes images with either OpenAI or Huggingface pretrained CLIP.
    https://huggingface.co/openai/clip-vit-large-patch14
    """
    if isinstance(clip_model, CLIPVisionModel):
        last_hidden_states = []

        for y in tqdm(y_list, desc="Preprocessing & encoding images"):
            model_input = preprocess(images=Image.open(y), return_tensors="pt")

            model_output = clip_model(**model_input.to(device))

            last_hidden_states.append(model_output.last_hidden_state.cpu())

        return torch.cat(last_hidden_states, dim=0)
    else:
        model_input = torch.stack(
            [
                preprocess(Image.open(y).convert("RGB"))
                for y in tqdm(y_list, desc="Preprocessing images")
            ]
        )

        return sequential_apply(
            model_input,
            clip_model.encode_image,
            batch_size=32,
            device=device,
            desc="Encoding images",
        ).float()


def encode_images_huggingface(
    y_list: List[str], preprocess, clip_model, device
) -> torch.Tensor:
    Y = torch.stack(
        [
            preprocess(Image.open(y).convert("RGB"))
            for y in tqdm(y_list, desc="Preprocessing images")
        ]
    )

    Y = sequential_apply(
        Y,
        clip_model.encode_image,
        batch_size=32,
        device=device,
        desc="Encoding images",
    ).float()

    return Y


def make_refined_split_from_file(epochs: mne.Epochs, category_idxs: np.ndarray):
    """Old version of make_split."""
    # NOTE: Event ids in the preprocessed data seem to start from 1.
    trials = epochs.events[:, -1] - 1  # ( 27048, )

    # NOTE: There can be some missing events as they were not presented to the subject.
    events, event_counts = np.unique(trials, return_counts=True)  # ( 22248 + 1, )
    # NOTE: 1 is for train images, 12 is for test images, and 2400 is for fixation.
    assert np.array_equal(np.unique(event_counts), [1, 12, 2400])

    test_events = np.take(events, np.where(event_counts == 12)[0])  # ( 200, )
    assert len(test_events) == 200

    test_categories = category_idxs[test_events]  # ( 200, )
    assert len(test_categories) == 200 & len(np.unique(test_categories)) == 200

    train_event_idxs = np.where(
        np.logical_not(np.isin(category_idxs, test_categories))
    )[0]
    test_event_idxs = np.where(np.isin(category_idxs, test_categories))[0]
    assert len(train_event_idxs) + len(test_event_idxs) == len(category_idxs)

    train_trial_idxs = np.where(np.isin(trials, train_event_idxs))[0]
    test_trial_idxs = np.where(np.isin(trials, test_event_idxs))[0]
    assert len(train_trial_idxs) + len(test_trial_idxs) + 2400 == len(epochs)

    return train_trial_idxs, test_trial_idxs


@hydra.main(
    version_base=None, config_path="../../configs/thingsmeg", config_name="clip"
)
def run(args: DictConfig) -> None:
    meg_paths = [
        os.path.join(args.meg_dir, f"preprocessed_P{i+1}-epo.fif") for i in range(4)
    ]
    sample_attrs_paths = [
        os.path.join(args.thingsmeg_dir, f"sourcedata/sample_attributes_P{i+1}.csv")
        for i in range(4)
    ]

    save_dir = os.path.join(args.preprocessed_data_dir, args.preproc_name)
    os.makedirs(save_dir, exist_ok=True)

    for subject_id, (meg_path, sample_attrs_path) in enumerate(zip(meg_paths, sample_attrs_paths)):  # fmt: skip
        cprint(f"==== Processing subject {subject_id+1} ====", "cyan")

        sample_attrs = np.loadtxt(
            sample_attrs_path, dtype=str, delimiter=",", skiprows=1
        )

        device = f"cuda:{args.cuda_id}"

        # -----------------
        #        MEG
        # -----------------
        if not args.skip_meg:
            cprint("> Loading epochs...", "cyan")
            epochs = mne.read_epochs(meg_path)

            cprint(f"> Resampling epochs to {args.brain_resample_sfreq}Hz...", "cyan")
            epochs.resample(args.brain_resample_sfreq, n_jobs=8)

            cprint(f"> Scale and clamping epochs to ±{args.clamp_lim}...", "cyan")
            epochs.apply_function(
                partial(scale_clamp, scale_transposed=False, clamp_lim=args.clamp_lim),
                n_jobs=8,
            )

            cprint("> Baseline correction...", "cyan")
            epochs.apply_baseline((None, 0))

            X = torch.from_numpy(epochs.get_data()).to(torch.float32)
            # ( 27048, 271, segment_len )

            cprint(f"MEG P{subject_id+1}: {X.shape}", "cyan")

            torch.save(X, os.path.join(save_dir, f"MEG_P{subject_id+1}.pt"))

        # -----------------
        #      Images
        # -----------------
        y_list = []
        for path in sample_attrs[:, 8]:
            if "images_meg" in path:
                y_list.append(
                    os.path.join(args.images_dir, "/".join(path.split("/")[1:]))
                )
            elif "images_test_meg" in path:
                y_list.append(
                    os.path.join(
                        args.images_dir,
                        "_".join(os.path.basename(path).split("_")[:-1]),
                        os.path.basename(path),
                    )
                )
            elif "images_catch_meg" in path:
                y_list.append(os.path.join(args.images_dir, "black.jpg"))
            else:
                raise ValueError(f"Unknown image path type: {path}")

        if args.vision.pretrained and not args.skip_images:
            np.savetxt(
                os.path.join(save_dir, f"Images_P{subject_id+1}.txt"),
                y_list,
                fmt="%s",
                delimiter="\n",
            )

            if subject_id == 0:
                if args.vision.pretrained_model.startswith("ViT-"):
                    clip_model, preprocess = clip.load(args.vision.pretrained_model)
                    clip_model = clip_model.eval().to(device)

                elif args.vision.pretrained_model.startswith("openai/"):
                    clip_model = CLIPVisionModel.from_pretrained(args.vision.pretrained_model).to(device)  # fmt: skip
                    preprocess = AutoProcessor.from_pretrained(
                        args.vision.pretrained_model
                    )
                else:
                    raise ValueError(f"Unknown pretrained CLIP type: {args.vision.pretrained_model}")  # fmt: skip

            Y = encode_images(y_list, preprocess, clip_model, device)

            cprint(f"Images P{subject_id+1}: {Y.shape}", "cyan")

            torch.save(Y, os.path.join(save_dir, f"Images_P{subject_id+1}.pt"))

        # -----------------
        #      Texts
        # -----------------
        if not args.skip_texts:
            categories = np.loadtxt(
                os.path.join(args.things_dir, "things_concepts.tsv"),
                dtype=str,
                delimiter="\t",
                skiprows=1,
                usecols=0,
            )
            categories = [
                categories[i] if i != 1854 else ""
                for i in sample_attrs[:, 2].astype(int) - 1
            ]

            if subject_id == 0:
                clip_text = uvit_clip.FrozenCLIPEmbedder()
                clip_text.eval()
                clip_text.to(device)

                if args.caption:
                    caption_model = "Salesforce/blip-image-captioning-large"
                    processor = BlipProcessor.from_pretrained(caption_model)
                    model = BlipForConditionalGeneration.from_pretrained(caption_model).to(device)  # fmt: skip

                    # NOTE: returns dict of captions sorted by image names.
                    texts_dict = caption_images(y_list, processor, model, device)

                    with open(os.path.join(save_dir, f"Captions.csv"), "w") as f:  # fmt: skip
                        writer = csv.writer(f)
                        writer.writerows([[t, c] for t, c in texts_dict.items()])

            if args.caption:
                texts = [
                    texts_dict[os.path.splitext(os.path.basename(y))[0]] for y in y_list
                ]
            else:
                texts = []
                for category in categories:
                    if category == "":
                        texts.append("")
                    else:
                        if category.startswith(("a", "e", "i", "o", "u")) and category not in ["unicycle", "uniform", "urinal"]:  # fmt: skip
                            texts.append(f"A photo of an {category}")
                        else:
                            texts.append(f"A photo of a {category}")

            contexts = torch.cat(
                [clip_text.encode(text) for text in tqdm(texts, desc="Encoding texts")]
            )

            cprint(f"Texts P{subject_id+1}: {contexts.shape}", "cyan")

            torch.save(contexts, os.path.join(save_dir, f"Texts_P{subject_id+1}.pt"))


if __name__ == "__main__":
    run()
