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

# ctx = multiprocessing.get_context("spawn")
# torch.multiprocessing.set_start_method("spawn", force=True)

from nd.utils.brain_preproc import brain_preproc
from nd.utils.face_preproc import FacePreprocessor
from nd.utils.stylegan_encoder import StyleGANEncoder
from nd.utils.preproc_utils import (
    get_face2brain_data_paths,
    crop_and_segment,
    h5_save,
)
from nd.utils.gTecUtils.gtec_preproc import eeg_subset_fromTrigger

from encoder4editing.utils.alignment import align_face


class FaceStyleGANPreprocessor(FacePreprocessor):
    def __init__(self, args, video_path: str, video_times_path: str) -> None:
        super().__init__(args, video_path)

        self.stylegan_encoder = StyleGANEncoder(args.stylegan.model_path)

        self.y_times = np.loadtxt(video_times_path, delimiter=",")  # ( ~100000, )

    def __call__(self, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_
        Args:
            batch_size (int, optional): Batch size to run StyleGAN encoder.
        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        y_list = []
        transformed_list = []
        drop_list = []

        i = 0
        pbar = tqdm(total=self.num_frames)
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                cprint("Reached to the final frame", color="yellow")
                break

            transformed = None

            extracted = self.extractor.run(i, frame)

            if extracted is not None:
                aligned = align_face(extracted, self.stylegan_encoder.predictor)

                if aligned is not None:
                    transformed = self.stylegan_encoder.img_transforms(aligned)

            if transformed is None:
                extracted = np.zeros((*self.output_size, 3), dtype=np.uint8)
                transformed = torch.zeros(3, *self.output_size, dtype=torch.float32)

                # NOTE: Saving indexes of segments after segmenting
                drop_list.append(i // self.segment_len)
                cprint(f"No face at {i}.", "yellow")

            transformed_list.append(transformed)

            self.writer.write(extracted)

            if len(transformed_list) == batch_size:
                pbar.update(batch_size)

                _, latents = self.stylegan_encoder.run_on_batch(transformed_list)
                y_list.append(latents)
                transformed_list = []

            i += 1

        self.cap.release()

        if not len(transformed_list) == 0:
            _, latents = self.stylegan_encoder.run_on_batch(transformed_list)
            y_list.append(latents)

        # NOTE: Avoid drop_segments being float when drop_list is empty
        drop_segments = np.unique(drop_list).astype(np.int64)

        y = torch.cat(y_list).numpy()  # ( ~100000, 18, 512 )
        assert len(y_times) == len(y), "Number of samples for y, y_times is different."

        y = crop_and_segment(y, self.segment_len)  # ( ~1000, 90, 18, 512 )

        y_times = y_times[:: self.segment_len]

        return y, y_times, drop_segments


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

        data_dir = f"data/preprocessed/stylegan/{args.preproc_name}/S{i}/"
        os.makedirs(data_dir, exist_ok=True)

        video_path, video_times_path, eeg_path = paths

        if os.path.exists(data_dir + "face_before_crop.npy"):
            tmp = np.load(data_dir + "tmp.npz")
            Y = tmp["face_before_crop"]
            face_times = tmp["face_times"]
            drop_segments = tmp["drop_segments"]
        else:
            Y, face_times, drop_segments = FaceStyleGANPreprocessor(
                args, video_path, video_times_path
            )()
            np.savez(
                data_dir + "tmp",
                face_before_crop=Y,
                face_times=face_times,
                drop_segments=drop_segments,
            )

        if eeg_path.endswith(".hdf5"):  # ICA is not done
            X, eeg_times, _ = eeg_subset_fromTrigger(args, eeg_path)

        elif eeg_path.endswith(".npz"):  # ICA is done
            _d = np.load(eeg_path)
            X, eeg_times = _d["eeg"], _d["times"]
            del _d

        else:
            raise NotImplementedError

        X, face_drops_prev, face_drops_after = brain_preproc(
            args,
            brain=X,
            segment_len=int(args.brain_resample_sfreq * args.seq_len),
            brain_times=eeg_times,
            face_times=face_times,
        )

        if face_drops_after == 0:
            cprint("No drops after", color="yellow")
            Y = Y[face_drops_prev:]
        else:
            Y = Y[face_drops_prev:-face_drops_after]

        assert len(X) == len(
            Y
        ), "Brain and face data have different number of segments."
        cprint(f"Session {i} EEG: {X.shape}", color="cyan")
        cprint(f"Session {i} face: {Y.shape}", color="cyan")

        X = np.delete(X, drop_segments, axis=0)
        Y = np.delete(Y, drop_segments, axis=0)

        cprint(f"Session {i} EEG (after drop): {X.shape}", color="cyan")
        cprint(f"Session {i} face (after drop): {Y.shape}", color="cyan")

        np.save(data_dir + "brain.npy", X)
        np.save(data_dir + "eeg_times.npy", eeg_times)
        h5_save(data_dir + "face.h5", Y)


if __name__ == "__main__":
    main()
