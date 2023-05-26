import os, sys
import numpy as np
import torch
import cv2
from termcolor import cprint
from tqdm import tqdm
from typing import Tuple
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, open_dict
import multiprocessing

ctx = multiprocessing.get_context("spawn")
torch.multiprocessing.set_start_method("spawn", force=True)

from brain2face.utils.brain_preproc import brain_preproc
from brain2face.utils.extractor import FaceExtractor
from brain2face.utils.stylegan_encoder import StyleGANEncoder
from brain2face.utils.preproc_utils import get_arayadriving_dataset_paths, export_gif
from brain2face.utils.gTecUtils.gtec_preproc import eeg_subset_fromTrigger
from brain2face.constants import EXTRACTED_VIDEO_ROOT

from encoder4editing.utils.alignment import align_face


def face_preproc(args, video_path, video_times_path) -> Tuple[torch.Tensor, np.ndarray]:
    # -------------------------
    #        Input Video
    # -------------------------
    cap = cv2.VideoCapture(video_path)
    input_size = np.array(
        [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))]
    )
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # -------------------------
    #      Face Extractor
    # -------------------------
    face_extractor = FaceExtractor(args.face_extractor, input_size, "center", fps)

    # -------------------------
    #         StyleGAN
    # -------------------------
    stylegan_encoder = StyleGANEncoder(args.stylegan.model_path)

    # -------------------------
    #         Outputs
    # -------------------------
    fmt = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(
        f"{EXTRACTED_VIDEO_ROOT}/{Path(video_path).stem}.mp4",
        fmt,
        fps,
        tuple(args.face_extractor.output_size),
    )

    y_times = np.loadtxt(video_times_path, delimiter=",")  # ( 109800, )

    y_list = []
    no_face_idxs = []
    i = 0
    batch_size = 32
    transformed_list = []

    segment_len = args.seq_len * fps  # 90

    if not args.debug:
        pbar = tqdm(total=num_frames)
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                cprint("Reached to the final frame", color="yellow")
                break

            extracted = face_extractor.run(i, frame)

            if extracted is not None:
                writer.write(extracted)

                aligned_image = align_face(extracted, stylegan_encoder.predictor)

                if aligned_image is not None:
                    transformed_image = stylegan_encoder.img_transforms(aligned_image)

                    transformed_list.append(transformed_image)

                else:
                    if i < len(y_times):
                        no_face_idxs.append(i)
            else:
                if i < len(y_times):
                    no_face_idxs.append(i)

                writer.write(
                    np.zeros((*args.face_extractor.output_size, 3), dtype=np.uint8)
                )

            i += 1
            if i % 100 == 0:
                pbar.update(100)

            if len(transformed_list) == batch_size:
                _, latents = stylegan_encoder.run_on_batch(transformed_list)
                y_list.append(latents)
                transformed_list = []

            # if i == 5000:
            #     break

            assert len(transformed_list) <= batch_size

        if not len(transformed_list) == 0:
            _, latents = stylegan_encoder.run_on_batch(transformed_list)
            y_list.append(latents)

        cap.release()

        y = torch.cat(y_list).numpy()  # ( 109797, 18, 512 )

    else:
        y = np.random.rand(100000, 18, 512)

    y = y[: -(y.shape[0] % segment_len)]  # ( 109710, 18, 512 )
    y = y.reshape(-1, segment_len, *y.shape[-2:])  # ( 1219, 90, 18, 512 )

    y_times = np.delete(y_times, no_face_idxs)
    y_times = y_times[::segment_len][: y.shape[0]]  # ( 1219, )

    assert len(y) == len(y_times)

    return y, y_times


@hydra.main(version_base=None, config_path="../../configs", config_name="stylegan")
def main(args: DictConfig) -> None:
    with open_dict(args):
        args.root_dir = get_original_cwd()

    video_paths, video_times_paths, eeg_paths = get_arayadriving_dataset_paths(
        args.data_root, args.ica_data_root, args.start_subj, args.end_subj
    )

    for i, paths in enumerate(zip(video_paths, video_times_paths, eeg_paths)):
        cprint(f"Processing subject number {args.start_subj + i}", color="cyan")

        video_path, video_times_path, eeg_path = paths

        Y, face_times = face_preproc(args, video_path, video_times_path)

        if eeg_path.endswith(".hdf5"):  # ICA is not done
            X, eeg_times, _ = eeg_subset_fromTrigger(args, eeg_path)

        elif eeg_path.endswith(".npz"):  # ICA is done
            _d = np.load(eeg_path)
            X, eeg_times = _d["eeg"], _d["times"]
            del _d

        else:
            raise NotImplementedError

        X, face_drops_prev, face_drops_after = brain_preproc(
            args, X, brain_times=eeg_times, face_times=face_times
        )
        cprint(f"Subject {i} EEG: {X.shape}", color="cyan")

        if face_drops_after == 0:
            cprint("No drops after", color="yellow")
            Y = Y[face_drops_prev:]
        else:
            Y = Y[face_drops_prev:-face_drops_after]

        cprint(f"Subject {i} face: {Y.shape}", color="cyan")

        assert len(X) == len(Y)

        data_dir = f"data/{args.preproc_name}/S{i}/"
        os.makedirs(data_dir, exist_ok=True)
        np.save(data_dir + "brain.npy", X)
        np.save(data_dir + "face.npy", Y)
        np.save(data_dir + "eeg_times.npy", eeg_times)
        np.save(data_dir + "face_times.npy", face_times)


if __name__ == "__main__":
    main()
