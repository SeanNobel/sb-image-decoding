import mne
import numpy as np
import torch
from brain2face.utils.gTecUtils.gTecUtils import loadMontage
from brain2face.constants import MONTAGE_INFO_PATH


def ch_locations_2d(args, training=True):
    if args.dataset == "ArayaDriving":
        montage = loadMontage(MONTAGE_INFO_PATH)
        info = mne.create_info(ch_names=montage.ch_names, sfreq=250.0, ch_types="eeg")
        info.set_montage(montage)

        layout = mne.channels.find_layout(info, ch_type="eeg")

        loc = layout.pos[:, :2]  # ( 32, 2 )

    else:
        raise ValueError()

    if training:
        # min-max normalization
        loc = (loc - loc.min(axis=0)) / (loc.max(axis=0) - loc.min(axis=0))

        return torch.from_numpy(loc.astype(np.float32))

    else:
        return info
