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
from brain2face.utils.face_preproc import FacePreprocessor
from brain2face.utils.stylegan_encoder import StyleGANEncoder
from brain2face.utils.preproc_utils import get_face2brain_data_paths, crop_and_segment
from brain2face.utils.gTecUtils.gtec_preproc import eeg_subset_fromTrigger

from encoder4editing.utils.alignment import align_face


class FaceStyleGANPreprocessor(FacePreprocessor):
    def __init__(self, args, video_path: str, video_times_path: str) -> None:
        super().__init__(args, video_path)

        self.stylegan_encoder = StyleGANEncoder(args.stylegan.model_path)

        self.y_times = np.loadtxt(video_times_path, delimiter=",")  # ( ~100000, )

    def __call__(self, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        y_list = []
        no_face_idxs = []
        transformed_list = []

        i = 0
        pbar = tqdm(total=self.num_frames)
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                cprint("Reached to the final frame", color="yellow")
                break

            transformed_image = None

            extracted = self.extractor.run(i, frame)

            if extracted is not None:
                aligned_image = align_face(extracted, self.stylegan_encoder.predictor)

                if aligned_image is not None:
                    transformed_image = self.stylegan_encoder.img_transforms(
                        aligned_image
                    )

                    transformed_list.append(transformed_image)

            else:
                extracted = np.zeros((*self.output_size, 3), dtype=np.uint8)

            if transformed_image is None and i < len(self.y_times):
                no_face_idxs.append(i)

            self.writer.write(extracted)

            if len(transformed_list) == batch_size:
                pbar.update(batch_size)

                _, latents = self.stylegan_encoder.run_on_batch(transformed_list)
                y_list.append(latents)
                transformed_list = []

            assert len(transformed_list) <= batch_size

            i += 1

        if not len(transformed_list) == 0:
            _, latents = self.stylegan_encoder.run_on_batch(transformed_list)
            y_list.append(latents)

        self.cap.release()

        y = torch.cat(y_list).numpy()  # ( ~100000, 18, 512 )
        y = crop_and_segment(y, self.segment_len)  # ( ~1000, 90, 18, 512 )

        y_times = np.delete(self.y_times, no_face_idxs)
        y_times = y_times[:: self.segment_len][: y.shape[0]]  # ( ~1000, )
        assert len(y_times) == len(y)

        return y, y_times


@hydra.main(
    version_base=None, config_path="../../configs/stylegan", config_name="ica_deep"
)
def main(args: DictConfig) -> None:
    with open_dict(args):
        args.root_dir = get_original_cwd()

    video_paths, video_times_paths, eeg_paths = get_face2brain_data_paths(
        args.data_root, args.ica_data_root, args.start_subj, args.end_subj
    )

    for _i, paths in enumerate(zip(video_paths, video_times_paths, eeg_paths)):
        i = args.start_subj + _i
        cprint(f"Processing subject number {i}", color="cyan")

        data_dir = f"data/{args.preproc_name}/S{i}/"
        os.makedirs(data_dir, exist_ok=True)

        video_path, video_times_path, eeg_path = paths

        if os.path.exists(data_dir + "face_before_crop.npy"):
            Y = np.load(data_dir + "face_before_crop.npy")
            face_times = np.load(data_dir + "face_times.npy")
        else:
            Y, face_times = FaceStyleGANPreprocessor(args, video_path, video_times_path)()
            np.save(data_dir + "face_before_crop.npy", Y)
            np.save(data_dir + "face_times.npy", face_times)

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

        np.save(data_dir + "brain.npy", X)
        np.save(data_dir + "face.npy", Y)
        np.save(data_dir + "eeg_times.npy", eeg_times)


if __name__ == "__main__":
    main()
