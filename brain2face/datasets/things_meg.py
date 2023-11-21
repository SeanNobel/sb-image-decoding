import os, sys
import torch
import mne
from termcolor import cprint
from time import time


class ThingsMEGCLIPDataset(torch.utils.data.Dataset):
    def __init__(self, args, train: bool = True) -> None:
        super().__init__()

        # Load preprocessed data in Hebart et al. (2023). There are slight differences from Meta paper.
        # It looks that scaling and baseline correction is performed but clamping is not.
        cprint("Loading epochs...", "cyan")
        
        meg_paths = [
            os.path.join(
                args.data_root, f"derivatives/preprocessed/preprocessed_P{i+1}-epo.fif"
            )
            for i in range(1)
        ]
        epochs = [mne.read_epochs(path) for path in meg_paths]

        # Resample from 200Hz to 120Hz as in Meta paper.
        cprint("Resampling epochs...", "cyan")
        stime = time()
        
        epochs = [epoch.resample(args.brain_resample_sfreq, n_jobs=24) for epoch in epochs]
        
        cprint(f"Done. ({time() - stime:.2f}s)", "cyan")

        X = [torch.from_numpy(epoch.get_data()) for epoch in epochs]

        # Image ID in the dataset.
        y = [torch.from_numpy(epoch.events[:, 2]) for epoch in epochs]

        subject_idxs = [torch.ones_like(y_) * i for i, y_ in enumerate(y)]

        print(X[0].shape, y[0].shape, subject_idxs[0].shape)

        sys.exit()
