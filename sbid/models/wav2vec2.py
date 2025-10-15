import sys
import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional, Union, List
from termcolor import cprint

from transformers import Wav2Vec2ConformerModel, Wav2Vec2ConformerConfig

from sbid.models.brain_encoder import SubjectBlock, BrainEncoderBase
from sbid.utils.layout import ch_locations_2d, DynamicChanLoc2d


class Wav2Vec2ConformerSpatialMixer(BrainEncoderBase):
    def __init__(self, args, subjects: Union[int, List[str]]) -> None:
        super().__init__()

        self.ignore_subjects = args.ignore_subjects

        num_subjects: int = subjects if isinstance(subjects, int) else len(subjects)
        layout: Union[ch_locations_2d, DynamicChanLoc2d] = eval(args.layout)

        assert layout == ch_locations_2d, "Supporting static spatial attention only for now."  # fmt: skip
        self.subject_block = SubjectBlock(
            args, num_subjects if not self.ignore_subjects else 1, layout(args)
        )

        # config = Wav2Vec2ConformerConfig()
        self.wav2vec2 = Wav2Vec2ConformerModel.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")  # fmt: skip

    def forward(self, X: torch.Tensor, subject_idxs: torch.Tensor) -> torch.Tensor:
        X = self.subject_block(
            X, subject_idxs if not self.ignore_subjects else torch.zeros_like(subject_idxs)  # fmt: skip
        )
        # ( b, D1, t )

        with torch.no_grad():
            X = rearrange(X, "b d t -> (b d) t")
            X = self.wav2vec2(X)

        print(X.shape)
        sys.exit()
