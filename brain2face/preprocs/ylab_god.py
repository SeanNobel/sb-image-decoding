import os, sys
import numpy as np
import mne
import h5py
import glob
from natsort import natsorted
from tqdm import tqdm
from termcolor import cprint
from typing import List, Tuple, Optional
import hydra
from omegaconf import DictConfig, open_dict

from brain2face.utils.brain_preproc import brain_preproc, baseline_correction
from brain2face.utils.layout import load_god_montage


def get_stim_fnames(ecog: h5py._hl.files.File) -> list:
    """
    Returns:
        Y: ( n_frames, )
    """
    Y = []
    
    stim_fname = ecog["trigger_info"]["stim_fname"]
    print(f"Number of stimuli: {len(stim_fname)}")
    
    for i in range(len(stim_fname)):
        ref = stim_fname[i][0] # h5py reference object
        
        # fname = ecog[ref][()].tobytes().decode("utf-8")
        fname = "".join([chr(c) for c in ecog[ref][()].squeeze()])
        
        # FIXME: empty string (not a double spaces but it looks like that!)
        # is automatically deleted when np.savetxt.
        if len(fname) <= 2:
            fname = "blank"
        
        Y.append(fname)
    
    return np.array(Y)

def get_segmented_ecog(args, ecog: h5py._hl.files.File) -> Tuple[np.ndarray, List[int]]:
    """
    Returns:
        X: ( frames, channels, timesteps )
    """    
    signals = ecog["data_st"]["signals"][()].astype(np.float64)
    # ( channels, timesteps )
    orig_sfreq = int(ecog["data_st"]["sampling_rate"][()][0][0]) # 10000Hz
    
    signals = brain_preproc(
        args, signals, orig_sfreq=orig_sfreq, segment=False, resample=False
    )
    
    onsets = ecog["trigger_info"]["stim_onset"][()].squeeze()
    onsets = np.nan_to_num(onsets, nan=-1).astype(int)
    offsets = ecog["trigger_info"]["stim_offset"][()].squeeze()
    offsets = np.nan_to_num(offsets, nan=-1).astype(int)
    print(f"Number of onsets: {len(onsets)}")
    
    X = []
    dropped_idxs = []
    for i, (onset, offset) in enumerate(zip(onsets, offsets)):
        
        if onset == -1: # or offset == -1:
            cprint("Dropped: onset was NaN", "yellow") # or offset was NaN", "yellow")
            dropped_idxs.append(i)
            continue
        
        chunk = signals[:, onset:onset + int(orig_sfreq * args.max_seq_len)]
        
        chunk = mne.filter.resample(chunk, down=orig_sfreq/args.brain_resample_sfreq)
        
        X.append(chunk)
                
    X = np.stack(X) # ( segments, channels, segment_len=250 )
    
    X = baseline_correction(
        X,
        int(args.max_seq_len * args.baseline_ratio * args.brain_resample_sfreq)
    )
    
    return X, dropped_idxs

def preproc(args, subject_name: str, fnames: List[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        montage = load_god_montage(subject_name, args.freesurfer_dir)
    except FileNotFoundError:
        cprint(f"Subject {subject_name} was dropped, as its montage was not found.", "yellow")
        return None, None
    
    X = []
    Y = []
    for fname in fnames:        
        ecog = h5py.File(fname, "r")
        
        _X, dropped_idxs = get_segmented_ecog(args, ecog)
        # ( segments, channels, max_segment_len )
        
        if _X.shape[1] != montage.shape[0]:
            cprint(f"Subject {subject_name} was dropped, as the number of channels didn't match to its montage.", "yellow")
            return None, None
        
        # NOTE: hold images as paths
        _Y = get_stim_fnames(ecog)
        _Y = np.delete(_Y, dropped_idxs, axis=0)
        
        assert _X.shape[0] == _Y.shape[0]
        
        X.append(_X)
        Y.append(_Y)
        
    return np.concatenate(X), np.concatenate(Y)


@hydra.main(version_base=None, config_path="../../configs/ylab/god", config_name="clip")
def main(args: DictConfig) -> None:
    with open_dict(args):
        args.root_dir = hydra.utils.get_original_cwd()
        
    for ecog_dir in glob.glob(f"{args.data_root}continuous_10k/*/"):
        subject_name = ecog_dir.split('/')[-2]
        cprint(f"Processing subject {subject_name}.", "cyan")
        
        fnames_train = natsorted(glob.glob(f"{ecog_dir}*Trn*.mat"))
        X_train, Y_train = preproc(args, subject_name, fnames_train)
        
        if X_train is None:
            continue
        
        fnames_val = natsorted(glob.glob(f"{ecog_dir}*Val*.mat"))
        X_val, Y_val = preproc(args, subject_name, fnames_val)
        
        if X_val is None:
            continue
            
        cprint(f"Subject {subject_name} has {len(fnames_train)} train sessions and {len(fnames_val)} validation sessions \nECoG: train {X_train.shape}, val {X_val.shape} \nImage: train {Y_train.shape}, val {Y_val.shape}", "cyan")
        
        data_dir = os.path.join(args.root_dir, "data/preprocessed/ylab/god", args.preproc_name, subject_name)
        os.makedirs(data_dir, exist_ok=True)
        np.save(os.path.join(data_dir, "brain_train.npy"), X_train)
        np.save(os.path.join(data_dir, "brain_test.npy"), X_val)
        np.savetxt(os.path.join(data_dir, "image_train.txt"), Y_train, delimiter=",", fmt="%s")
        np.savetxt(os.path.join(data_dir, "image_test.txt"), Y_val, delimiter=",", fmt="%s")
        
if __name__ == "__main__":
    main()