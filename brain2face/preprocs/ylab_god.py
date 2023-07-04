import os, sys
import numpy as np
import mne
import h5py
import glob
from natsort import natsorted
from tqdm import tqdm
from termcolor import cprint
from typing import List, Tuple
import hydra
from omegaconf import DictConfig, open_dict

from brain2face.utils.brain_preproc import brain_preproc, baseline_correction

def get_stim_fnames(ecog: h5py._hl.files.File) -> list:
    """
    Returns:
        Y: ( n_frames, )
    """
    Y = []
    
    stim_fname = ecog["trigger_info"]["stim_fname"]
    
    for i in range(len(stim_fname)):
        ref = stim_fname[i][0] # h5py reference object
        fname = ecog[ref][()].tobytes().decode("utf-8")
        
        Y.append(fname)
    
    return np.array(Y)

def get_segmented_ecog(args, ecog: h5py._hl.files.File) -> Tuple[np.ndarray, List[int]]:
    """
    Returns:
        X: ( n_frames, n_channels, n_timesteps )
    """    
    signals = ecog["data_st"]["signals"][()].astype(np.float64)
    orig_sfreq = int(ecog["data_st"]["sampling_rate"][()][0][0])
    
    signals = brain_preproc(args, signals, orig_sfreq=orig_sfreq, segment=False, resample=False)
    
    onsets = ecog["trigger_info"]["stim_onset"][()].squeeze()
    onsets = np.nan_to_num(onsets, nan=-1).astype(int)
    offsets = ecog["trigger_info"]["stim_offset"][()].squeeze()
    offsets = np.nan_to_num(offsets, nan=-1).astype(int)
    
    X = []
    dropped_idxs = []
    for i, (onset, offset) in enumerate(zip(onsets, offsets)):
        
        if onset == -1 or offset == -1:
            cprint("Dropped: onset or offset was NaN", "yellow")
            dropped_idxs.append(i)
            continue
        
        chunk = signals[:, onset:offset]
        
        # NOTE: Stimulus presentation seems to be 0.5s
        if chunk.shape[1] < orig_sfreq * 0.5:
            cprint(f"Chunk was padded since it was shorter than 0.5s: {chunk.shape}", "yellow")
            
            chunk = np.pad(chunk, ((0, 0), (0, int(orig_sfreq * 0.5) - chunk.shape[1])), mode="edge")
        else:
            chunk = chunk[:, :int(orig_sfreq * 0.5)]
        
        chunk = mne.filter.resample(chunk, down=orig_sfreq/args.brain_resample_sfreq)
        
        X.append(chunk)
                
    X = np.stack(X) # ( segments, channels, segment_len=250 )
    
    X = baseline_correction(X, int(0.5 * args.baseline_ratio * args.brain_resample_sfreq))
    
    return X, dropped_idxs

def preproc(args, fnames: List[str]):
    X = []
    Y = []
    for fname in fnames:
        ecog = h5py.File(fname, "r")
        
        _X, dropped_idxs = get_segmented_ecog(args, ecog)
        
        # NOTE: hold images as paths
        _Y = get_stim_fnames(ecog)
        _Y = np.delete(_Y, dropped_idxs, axis=0)
        
        X.append(_X)
        Y.append(_Y)
        
    return np.concatenate(X), np.concatenate(Y)


@hydra.main(version_base=None, config_path="../../configs/ylab", config_name="god")
def main(args: DictConfig) -> None:
    with open_dict(args):
        args.root_dir = hydra.utils.get_original_cwd()
        
    for i, ecog_dir in enumerate(glob.glob(f"{args.ecog_data_root}*/")):
        cprint(f"Processing subject {ecog_dir.split('/')[-2]}.", "cyan")
        
        fnames_train = natsorted(glob.glob(f"{ecog_dir}*Trn*.mat"))
        X_train, Y_train = preproc(args, fnames_train)
        
        fnames_val = natsorted(glob.glob(f"{ecog_dir}*Val*.mat"))
        X_val, Y_val = preproc(args, fnames_val)
            
        cprint(f"Subject {i} ({ecog_dir.split('/')[-2]}) has {len(fnames_train)} train sessions and {len(fnames_val)} validation sessions | ECoG: train {X_train.shape}, val {X_val.shape} | Image: train {Y_train.shape}, val {Y_val.shape}", "cyan")
        
        data_dir = os.path.join(args.root_dir, "data/preprocessed/ylab/god", args.preproc_name, f"S{i}")
        os.makedirs(data_dir, exist_ok=True)
        np.savez(os.path.join(data_dir, "brain.npz"), train=X_train, val=X_val)
        np.savez(os.path.join(data_dir, "image.npz"), train=Y_train, val=Y_val)
        
if __name__ == "__main__":
    main()