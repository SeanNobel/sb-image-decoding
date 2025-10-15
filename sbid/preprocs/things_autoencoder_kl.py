import os, sys
import numpy as np
import torch
from PIL import Image
from termcolor import cprint
from tqdm import tqdm

from uvit import libs


def run() -> None:
    sample_attrs_paths = [
        f"/mnt/tsukuyomi/things-meg/sourcedata/sample_attributes_P{i+1}.csv" for i in range(4)
    ]

    save_dir_ = "/home/sensho/brain2face/data/preprocessed/thingsmeg/4_autoencoder_kl"

    images_dir = "/mnt/tsukuyomi/things/osfstorage/THINGS/Images/"

    device = "cuda:0"

    autoencoder = libs.autoencoder.get_model(
        "uvit/assets/stable-diffusion/autoencoder_kl.pth"  # , scale_factor=0.23010
    ).to(device)

    for subject_id, sample_attrs_path in enumerate(sample_attrs_paths):
        cprint(f"==== Processing subject {subject_id+1} ====", "cyan")
        save_dir = os.path.join(save_dir_, f"Image_moments_P{subject_id+1}")
        os.makedirs(save_dir, exist_ok=True)

        sample_attrs = np.loadtxt(sample_attrs_path, dtype=str, delimiter=",", skiprows=1)

        # -----------------
        #      Images
        # -----------------
        y_list = []
        for path in sample_attrs[:, 8]:
            if "images_meg" in path:
                y_list.append(os.path.join(images_dir, "/".join(path.split("/")[1:])))
            elif "images_test_meg" in path:
                y_list.append(
                    os.path.join(
                        images_dir,
                        "_".join(os.path.basename(path).split("_")[:-1]),
                        os.path.basename(path),
                    )
                )
            elif "images_catch_meg" in path:
                y_list.append(os.path.join(images_dir, "black.jpg"))
            else:
                raise ValueError(f"Unknown image path type: {path}")

        np.savetxt(
            os.path.join(save_dir, f"Images_P{subject_id+1}.txt"),
            y_list,
            fmt="%s",
            delimiter="\n",
        )

        for i, y in enumerate(tqdm(y_list, desc="Preprocessing & encoding images")):
            image = Image.open(y).convert("RGB")
            image = image.resize((256, 256), Image.LANCZOS)
            image = np.array(image, dtype=np.float32) / 127.5 - 1.0
            image = torch.from_numpy(image).permute(2, 0, 1)

            moment = autoencoder.encode_moments(image.unsqueeze(0).to(device))

            np.save(os.path.join(save_dir, f"{i}.npy"), moment.cpu().numpy())

        cprint(f"Saved {i + 1} image moments ({moment.shape}) for P{subject_id+1}", "cyan")  # fmt: skip


if __name__ == "__main__":
    run()
