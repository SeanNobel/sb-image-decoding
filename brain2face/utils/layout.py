import os
import mne
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from glob import glob
from natsort import natsorted
from typing import Union, List

from brain2face.utils.gTecUtils.gTecUtils import loadMontage


class DynamicChanLoc2d:
    def __init__(self, args, subject_names: List[str]) -> None:
        if args.dataset == "YLabGOD":
            locations = [
                load_god_montage(subject, args.freesurfer_dir) for subject in subject_names
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


def load_god_montage(subject: str, freesurfer_dir: str) -> np.ndarray:
    path = os.path.join(freesurfer_dir, subject, "elec_recon", f"{subject}.DURAL")
        
    montage = np.loadtxt(path, delimiter=" ", skiprows=2, dtype="unicode").astype(float)
    montage = np.stack(montage) # ( n_channels, 3 )

    return montage


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
