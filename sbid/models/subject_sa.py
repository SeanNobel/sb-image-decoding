import numpy as np
import torch
import torch.nn as nn

from nd.utils.layout import DynamicChanLoc2d


class SubjectSpatialAttention(nn.Module):
    def __init__(self, args, loc: np.ndarray, SpatialAttention):
        super().__init__()

        self.num_channels = loc.shape[0]

        self.spatial_attention = SpatialAttention(args, loc)

        self.conv = nn.Conv1d(
            in_channels=args.D1,
            out_channels=args.D1,
            kernel_size=1,
            stride=1,
            bias=True,  # args.biases.conv_subj_sa,
        )

    def forward(self, X):
        """
        Args:
            X: ( 1, channels (+pad), timesteps )
        """
        X, pad = torch.split(
            X, [self.num_channels, X.shape[1] - self.num_channels], dim=1
        )
        assert pad.sum() == 0

        X = self.spatial_attention(X)

        X = self.conv(X)
        # X = self.conv2(X)

        return X


class SubjectBlockConvDynamic(nn.Module):
    def __init__(self, args, num_subjects: int, layouts: DynamicChanLoc2d) -> None:
        super().__init__()

        self.num_subjects = num_subjects
        self.num_channels = [
            layouts.get_loc(i).shape[0] for i in range(self.num_subjects)
        ]

        self.subject_layer = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.num_channels[i],
                    out_channels=args.D1,
                    kernel_size=1,
                    bias=False,
                    stride=1,
                )
                for i in range(self.num_subjects)
            ]
        )

    def forward(self, X: torch.Tensor, subject_idxs: torch.Tensor) -> torch.Tensor:
        X = torch.cat(
            [
                self.subject_layer[i](
                    torch.split(
                        x,
                        [self.num_channels[i], x.shape[0] - self.num_channels[i]],
                        dim=0,
                    )[0].unsqueeze(dim=0)
                )
                for i, x in zip(subject_idxs, X)
            ]
        )

        return X


class SubjectBlockSA(nn.Module):
    """Applies Spatial Attention to each subject separately"""

    def __init__(self, args, num_subjects: int, layouts: DynamicChanLoc2d) -> None:
        super().__init__()

        self.layouts = layouts
        self.num_subjects = num_subjects

        self.subject_layer = nn.ModuleList(
            [
                SubjectSpatialAttention(args, self.layouts.get_loc(i))
                for i in range(self.num_subjects)
            ]
        )

        self.conv = nn.Conv1d(
            in_channels=args.D1,
            out_channels=args.D1,
            kernel_size=1,
            stride=1,
            bias=True,  # args.biases.conv_block,
        )

    def forward(
        self,
        X: torch.Tensor,
        subject_idxs: torch.Tensor,
        subbatch: bool = False,
    ) -> torch.Tensor:
        """Currently SubjectBlockSA doesn't allow unknown subject.
        TODO: think of how to incorporate unknown layout.
        NOTE: X dim=1 is zero-padded depending on its original channel numbers.
        FIXME: inputting samples with batch_size=1 might make learning unstable.
        """
        if subbatch:
            X = torch.cat(
                [
                    self.subject_layer[i](X[subject_idxs == i])
                    for i in range(self.num_subjects)
                ]
            )

            regather_idxs = []
            for i in range(len(subject_idxs)):
                prev, after = torch.tensor_split(subject_idxs, [i])
                after = after[1:]
                regather_idxs.append(
                    (prev <= subject_idxs[i]).sum() + (after < subject_idxs[i]).sum()
                )
            regather_idxs = torch.tensor(regather_idxs).to(X.device)

            X = torch.index_select(X, dim=0, index=regather_idxs)

        else:
            # Sequential batch size = 1 input to each subject layer
            X = torch.cat(
                [
                    self.subject_layer[i](x.unsqueeze(dim=0))
                    for i, x in zip(subject_idxs, X)
                ]
            )

        X = self.conv(X)

        return X
