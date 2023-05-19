import mne
import numpy as np
import torch
from typing import Union

from brain2face.utils.gTecUtils.gTecUtils import loadMontage
from brain2face.constants import MONTAGE_INFO_PATH


def ch_locations_2d(args, training=True) -> Union[torch.Tensor, mne.Info]:
    if args.dataset == "Brain2FaceStyleGANDataset":
        montage = loadMontage(MONTAGE_INFO_PATH)
        info = mne.create_info(ch_names=montage.ch_names, sfreq=250.0, ch_types="eeg")
        info.set_montage(montage)

        if not training:
            return info

        layout = mne.channels.find_layout(info, ch_type="eeg")

        loc = layout.pos[:, :2]  # ( 32, 2 )

    elif args.dataset == "Brain2FaceYLabECoGDataset":
        # FIXME: correct later
        loc = np.meshgrid(np.arange(7), np.arange(10))
        loc = np.stack(loc).reshape(2, -1).T  # ( 70, 2 )

    else:
        raise ValueError()

    # min-max normalization
    loc = (loc - loc.min(axis=0)) / (loc.max(axis=0) - loc.min(axis=0))

    # NOTE: "In practice, as a_j is periodic, we scale down (x,y) to keep a margin of 0.1 on each side."
    loc = loc * 0.8 + 0.1

    return torch.from_numpy(loc.astype(np.float32))
