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

def get_segmented_ecog(
    args, ecog: h5py._hl.files.File
) -> Tuple[np.ndarray, List[int]]:
    """
    Returns:
        X: ( frames, channels, timesteps )
    """    
    signals = ecog["data_st"]["signals"][()].astype(np.float64)
    # ( channels, timesteps )
    orig_sfreq = int(ecog["data_st"]["sampling_rate"][()][0][0]) # 10000Hz
    
    signals = brain_preproc(
        args,
        signals,
        orig_sfreq=orig_sfreq,
        segment=False,
        resample=False,
        clamp=False,
        notch=[60, 120, 180, 240],
    )
    
    onsets = ecog["trigger_info"]["stim_onset"][()].squeeze()
    onsets = np.nan_to_num(onsets, nan=-1).astype(int)
    offsets = ecog["trigger_info"]["stim_offset"][()].squeeze()
    offsets = np.nan_to_num(offsets, nan=-1).astype(int)
    print(f"Number of onsets: {len(onsets)}")
    
    X = []
    baseline = []
    dropped_idxs = []
    dropped_idxs_clamp = []
    for i, (onset, offset) in enumerate(zip(onsets, offsets)):
        
        if onset == -1: # or offset == -1:
            cprint("Segment dropped: onset was NaN", "yellow") # or offset was NaN", "yellow")
            dropped_idxs.append(i)
            continue
        
        chunk = signals[:, onset:onset + int(orig_sfreq * args.max_seq_len)]
        
        if np.abs(chunk).max() > args.clamp_lim:
            cprint(f"Segment dropped: max value was {np.abs(chunk).max()}", "yellow")
            dropped_idxs_clamp.append(i)
            continue
        
        chunk = mne.filter.resample(chunk, down=orig_sfreq/args.brain_resample_sfreq)
        
        X.append(chunk)
        
        if args.baseline_ratio < 0:
            baseline.append(
                signals[:, onset - int(orig_sfreq * (-args.baseline_ratio) * args.max_seq_len):onset]
            )
            
    cprint(f"{len(dropped_idxs_clamp)} / {i} segments were dropped due to exceeding the clamp limit.", "yellow") # fmt: skip
                
    X = np.stack(X) # ( segments, channels, segment_len=250 )
    
    if len(baseline) > 0:
        X = baseline_correction(X, baseline=np.stack(baseline))
    else:
        X = baseline_correction(
            X,
            baseline_len_samp=int(args.max_seq_len * args.baseline_ratio * args.brain_resample_sfreq)
        )
        
    dropped_idxs += dropped_idxs_clamp
    dropped_idxs.sort()
    
    return X, dropped_idxs

def rename_channels(ch_names: List[str]) -> List[str]:
    """
    Rename channels to match the ECoG data.
    e.g.) 'A1-2' -> 'A2', 'A45-7' -> 'A51'
    """
    ch_names_new = []
    for name in ch_names:
        prefix, number = name[0], name[1:]
        block, number = number.split('-')
        number = str(int(block) + int(number) - 1)
        
        ch_names_new.append(prefix + number)
        
    return ch_names_new

def rearrange_montage(
    montage: np.ndarray,
    chnames_montage: List[str],
    ecog: h5py._hl.files.File
) -> Tuple[List]:
    """
    montage ( n_channels, 3 )
    """
    chnames_ecog = ["".join([chr(c) for c in ecog[ecog["data_st"]["channel_names"][i, 0]][()].squeeze()]) for i in range(len(ecog["data_st"]["channel_names"]))]
    
    montage_new = []
    exist = []
    for name in chnames_ecog:
        try:
            montage_new.append(montage[chnames_montage.index(name)])
            exist.append(True)
            
        except ValueError:
            exist.append(False)
            continue
        
    return montage_new, exist

def preproc(
    args, subject_name: str, fname: str
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        montage, ch_names = load_god_montage(subject_name, args.freesurfer_dir, return_chnames=True)
    except FileNotFoundError:
        cprint(f"Subject {subject_name} was dropped, as its montage was not found.", "yellow")
        return None

    ch_names = rename_channels(ch_names)
    if len(set(ch_names)) != len(ch_names):
        cprint(f"Subject {subject_name} was dropped, as its montage had duplicated channels.", "yellow")
        return None
    
    ecog = h5py.File(fname, "r")
    
    montage, exist = rearrange_montage(montage, ch_names, ecog)
    
    if len(montage) < 10:
        cprint(f"Subject {subject_name} was dropped, as the number of channels found was less than 10. ({len(montage)})", "yellow")
        return None
    
    montage = np.stack(montage)
            
    X, dropped_idxs = get_segmented_ecog(args, ecog)
    # ( segments, channels, max_segment_len )
    
    X = X[:, exist]
    assert X.shape[1] == montage.shape[0]
    
    # NOTE: hold images as paths
    Y = get_stim_fnames(ecog)
    Y = np.delete(Y, dropped_idxs, axis=0)
    
    assert X.shape[0] == Y.shape[0]
        
    return X, Y, montage


@hydra.main(version_base=None, config_path="../../configs/ylab/god", config_name="clip")
def main(args: DictConfig) -> None:
    with open_dict(args):
        args.root_dir = hydra.utils.get_original_cwd()
        
    for ecog_dir in glob.glob(f"{args.data_root}continuous_10k/*/"):
        subject_name = ecog_dir.split('/')[-2]
        data_dir = os.path.join(args.root_dir, "data/preprocessed/ylab/god", args.preproc_name, subject_name)
        
        fnames_trn = natsorted(glob.glob(f"{ecog_dir}*Trn*.mat"))
        fnames_val = natsorted(glob.glob(f"{ecog_dir}*Val*.mat"))
        fnames = fnames_trn + fnames_val
        cprint(f"Processing subject {subject_name}. ({len(fnames_trn)} train sessions, {len(fnames_val)} validation sessions)", "cyan")
        
        montages = [] # For asserting montage consistency
        for fname in fnames:
            preprocessed = preproc(args, subject_name, fname)
            if preprocessed is None:
                continue
            
            X, Y, montage = preprocessed
            montages.append(montage)
            
            # NOTE: Create here to skip empty subjects.
            if not os.path.exists(data_dir):
                os.makedirs(data_dir, exist_ok=True)
            
            postfix = os.path.splitext(os.path.basename(fname))[0].lower()
            np.save(os.path.join(data_dir, f"brain_{postfix}.npy"), X)
            np.savetxt(os.path.join(data_dir, f"image_{postfix}.txt"), Y, delimiter=",", fmt="%s")
            
        if len(montages) == 0:
            continue
            
        assert np.diff(np.stack(montages), axis=0).sum() == 0, "Montage inconsistency detected."
        np.save(os.path.join(data_dir, "montage.npy"), montages[0])
        
        # fnames_val = natsorted(glob.glob(f"{ecog_dir}*Val*.mat"))
        # X_val, Y_val = preproc(args, subject_name, fnames_val)
        
        # if X_val is None:
        #     continue
            
        # cprint(f"Subject {subject_name} has {len(fnames_train)} train sessions and {len(fnames_val)} validation sessions \nECoG: train {X_train.shape}, val {X_val.shape} \nImage: train {Y_train.shape}, val {Y_val.shape}", "cyan")

        
if __name__ == "__main__":
    main()