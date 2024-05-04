import os
import mne
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from glob import glob
from natsort import natsorted
from typing import Union, List, Tuple, Optional

from nd.utils.gTecUtils.gTecUtils import loadMontage


class DynamicChanLoc2d:
    def __init__(self, args, subject_names: List[str]) -> None:
        if args.dataset == "YLabGOD":
            locations = [
                np.load(
                    os.path.join(
                        "data/preprocessed/ylab/god",
                        args.preproc_name,
                        subject,
                        "montage.npy",
                    )
                )
                for subject in subject_names
            ]

            if args.loc_random:
                self.locations = [np.random.rand(*loc.shape) for loc in locations]

            else:
                num_channels = [loc.shape[0] for loc in locations]

                locations = np.concatenate(locations)
                locations = TSNE(n_components=2).fit_transform(locations)

                locations = min_max_norm(locations)

                self.locations = np.split(locations, np.cumsum(num_channels)[:-1])

        else:
            raise NotImplementedError

    def get_loc(self, subject_idx: int) -> torch.Tensor:
        loc = self.locations[subject_idx]

        return torch.from_numpy(loc.astype(np.float32))


def ch_locations_2d(args, training=True) -> Union[torch.Tensor, mne.Info]:
    if args.dataset == "YLabE0030":
        if args.loc_random:
            loc = np.meshgrid(np.arange(7), np.arange(10))
            loc = np.stack(loc).reshape(2, -1).T  # ( 70, 2 )
        else:
            loc = np.load(args.montage_path)

    elif args.dataset == "YLabGOD":
        assert args.loc_random, "Only implementing static YLabGOD for debug."
        loc = np.random.rand(48, 2)

    elif args.dataset == "UHD":
        montage = load_uhd_montage(args.montage_path)
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

    elif "ThingsMEG" in args.dataset:
        loc = np.load(args.montage_path)

    elif "ImageNetEEG" in args.dataset:
        # Copied and processed from https://github.com/perceivelab/eeg_visual_classification/issues/4
        channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'Fpz', 'F9', 'AFF5h', 'AFF1h', 'AFF2h', 'AFF6h', 'F10', 'FTT9h', 'FTT7h', 'FCC5h', 'FCC3h', 'FCC1h', 'FCC2h', 'FCC4h', 'FCC6h', 'FTT8h', 'FTT10h', 'TPP9h', 'TPP7h', 'CPP5h', 'CPP3h', 'CPP1h', 'CPP2h', 'CPP4h', 'CPP6h', 'TPP8h', 'TPP10h', 'POO9h', 'POO1', 'POO2', 'POO10h', 'Iz', 'AFp1', 'AFp2', 'FFT9h', 'FFT7h', 'FFC5h', 'FFC3h', 'FFC1h', 'FFC2h', 'FFC4h', 'FFC6h', 'FFT8h', 'FFT10h', 'TTP7h', 'CCP5h', 'CCP3h', 'CCP1h', 'CCP2h', 'CCP4h', 'CCP6h', 'TTP8h', 'P9', 'PPO9h', 'PPO5h', 'PPO1h', 'PPO2h', 'PPO6h', 'PPO10h', 'P10', 'I1', 'OI1h', 'OI2h', 'I2']  # fmt: skip
        assert len(channels) == 128

        montage = mne.channels.make_standard_montage("brainproducts-RNP-BA-128")
        positions = montage.get_positions()["ch_pos"]

        # Create missing channels as interpolation of surrounding channels
        positions["FTT9h"] = (positions["FT9"] + positions["T7"]) / 2
        positions["FTT10h"] = (positions["FT10"] + positions["T8"]) / 2
        positions["FFT9h"] = (positions["F7"] + positions["FT9"]) / 2
        positions["FFT10h"] = (positions["F8"] + positions["FT10"]) / 2

        loc = np.stack([positions[key] for key in channels])[:, :2]  # ( 128, 2 )
    else:
        raise ValueError()

    loc = min_max_norm(loc)

    return torch.from_numpy(loc.astype(np.float32))


def min_max_norm(loc: np.ndarray) -> np.ndarray:
    """Min-max normalization with margin of 0.1 on each side.
    Args:
        loc: ( num_channels, 2 )
    """
    loc = (loc - loc.min(axis=0)) / (loc.max(axis=0) - loc.min(axis=0))
    # NOTE: "In practice, as a_j is periodic, we scale down (x,y) to keep a margin of 0.1 on each side."
    loc = loc * 0.8 + 0.1

    return loc


def load_god_montage(
    subject: str, freesurfer_dir: str, return_chnames: bool = False
) -> Tuple[np.ndarray, Optional[List[str]]]:
    path = os.path.join(freesurfer_dir, subject, "elec_recon", f"{subject}.DURAL")
    montage = np.loadtxt(path, delimiter=" ", skiprows=2, dtype="unicode").astype(float)
    montage = np.stack(montage)  # ( n_channels, 3 )

    if not return_chnames:
        return montage

    path = os.path.join(freesurfer_dir, subject, "elec_recon", f"{subject}.electrodeNames")
    ch_names = [name[0] for name in np.loadtxt(path, delimiter=" ", skiprows=2, dtype="unicode")]

    return montage, ch_names


def load_uhd_montage(elec_filename):
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
