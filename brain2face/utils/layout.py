import os
import mne
import numpy as np
import torch
from glob import glob
from natsort import natsorted
from typing import Union

from brain2face.utils.gTecUtils.gTecUtils import loadMontage


def dynamic_ch_locations_2d(args, subject_idx: int) -> torch.Tensor:
    if args.dataset == "YLabGOD":
        montage_path = natsorted(glob(
            os.path.join(args.montage_dir, "random" if args.loc_random else "", "*.npy")
        ))[subject_idx]

        # FIXME: loading line
        # TODO: Normalize across subjects to align locations
        loc = np.load(montage_path)
        
    else:
        raise NotImplementedError

    # min-max normalization
    loc = (loc - loc.min(axis=0)) / (loc.max(axis=0) - loc.min(axis=0))

    # NOTE: "In practice, as a_j is periodic, we scale down (x,y) to keep a margin of 0.1 on each side."
    loc = loc * 0.8 + 0.1

    return torch.from_numpy(loc.astype(np.float32))


def ch_locations_2d(args, training=True) -> Union[torch.Tensor, mne.Info]:
    if args.dataset == "YLabE0030":
        if args.loc_random:
            loc = np.meshgrid(np.arange(7), np.arange(10))
            loc = np.stack(loc).reshape(2, -1).T  # ( 70, 2 )
        else:
            loc = np.load(args.montage_path)

    elif args.dataset == "UHD":
        montage = get_montage(args.montage_path)
        info = mne.create_info(
            ch_names=montage.ch_names, sfreq=args.brain_resample_sfreq, ch_types="eeg"
        )
        info.set_montage(montage)

        layout = mne.channels.find_layout(info, ch_type="eeg")

        # NOTE: First 128 channels out of 144 are the electrodes
        # TODO: Implement normalization for 3d
        # NOTE: Projects to xy plane (look at load_montage.ipynb)
        loc = layout.pos[: args.num_channels, :2]
        
    elif args.dataset == "StyleGAN":
        montage = loadMontage(args.montage_path)
        info = mne.create_info(ch_names=montage.ch_names, sfreq=250.0, ch_types="eeg")
        info.set_montage(montage)

        if not training:
            return info

        layout = mne.channels.find_layout(info, ch_type="eeg")

        loc = layout.pos[:, :2]  # ( 32, 2 )

    else:
        raise ValueError()

    # min-max normalization
    loc = (loc - loc.min(axis=0)) / (loc.max(axis=0) - loc.min(axis=0))

    # NOTE: "In practice, as a_j is periodic, we scale down (x,y) to keep a margin of 0.1 on each side."
    loc = loc * 0.8 + 0.1

    return torch.from_numpy(loc.astype(np.float32))


def get_montage(elec_filename):
    # Parse xml file for electrode position
    import xml.etree.ElementTree as ET

    tree = ET.parse(elec_filename)
    root = tree.getroot()

    elec_locs = {}
    nasion, lpa, rpa = None, None, None
    for child in root:
        elec = int(child.attrib["Id"])
        elec_loc = child.findtext("Positions/Subject/Head").split(",")
        elec_loc = [0.001 * float(el) for el in elec_loc]  # [mm] -> [m]
        assert 1 <= elec <= 147
        if elec <= 144:
            if elec <= 128:
                key = f"Ch-{elec:03}"
            else:
                key = f"Ext-{elec - 128:02}"
            elec_locs[key] = elec_loc
        else:
            if elec == 145:
                key = "nasion"
                nasion = elec_loc
            elif elec == 146:
                key = "lpa"
                lpa = elec_loc
            else:  # 147
                key = "rpa"
                rpa = elec_loc

    # generate montage
    mon = mne.channels.make_dig_montage(
        elec_locs, nasion=nasion, lpa=lpa, rpa=rpa, coord_frame="head"
    )

    return mon
