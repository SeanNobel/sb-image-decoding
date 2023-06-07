import os, sys
import numpy as np
import torch
import cv2
from termcolor import cprint
from tqdm import tqdm
from typing import Tuple, Optional
from pathlib import Path
import multiprocessing

ctx = multiprocessing.get_context("spawn")
torch.multiprocessing.set_start_method("spawn", force=True)

from brain2face.utils.extractor import FaceExtractor
from brain2face.utils.preproc_utils import crop_and_segment
from brain2face.constants import EXTRACTED_VIDEO_ROOT


class FacePreprocessor:
    def __init__(self, args, video_path: str) -> None:
        self.cap = cv2.VideoCapture(video_path)
        input_size = np.array(
            [
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            ]
        )
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.segment_len = self.fps * args.seq_len  # 90

        self.extractor = FaceExtractor(
            args.face_extractor, input_size, "center", self.fps
        )
        self.output_size = args.face_extractor.output_size

        fmt = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        self.writer = cv2.VideoWriter(
            f"{EXTRACTED_VIDEO_ROOT}/{Path(video_path).stem}.mp4",
            fmt,
            self.fps,
            tuple(args.face_extractor.output_size),
        )

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        """Processes the video and returns the extracted faces.
        Returns:
            y: ( segments, segment_len, 256, 256, 3 )
            drop_segments: Unique indexes of segments to be dropped later
        """
        y_list = []
        drop_list = []

        i = 0
        pbar = tqdm(total=self.num_frames)
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                cprint("Reached to the final frame", color="yellow")
                break

            extracted = self.extractor.run(i, frame)

            if extracted is None:
                extracted = np.zeros((*self.output_size, 3), dtype=np.uint8)

                # NOTE: Saving indexes of segments after segmenting
                drop_list.append(i // self.segment_len)
                cprint(f"No face at {i}.", "yellow")

            y_list.append(extracted)

            self.writer.write(extracted)

            i += 1

            if i % 1000 == 1:
                pbar.update(1000)

        self.cap.release()

        drop_segments = np.unique(drop_list)

        y = np.concatenate(y_list)  # ( ~100000, 256, 256, 3 )
        y = crop_and_segment(y, self.segment_len, drop_segments)
        # ( ~1000, 90, 256, 256, 3 )

        return y, drop_segments
