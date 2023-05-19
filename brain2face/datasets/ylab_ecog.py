import numpy as np
import pandas as pd
import h5py
from natsort import natsorted
import glob
import re
from termcolor import cprint
from tqdm import tqdm

import torch

class YLabECoGDataset(torch.utils.data.Dataset):
    def __init__(self, args, train: bool):
        super().__init__()
        self.ecog_data_root = args.ecog_data_root
        
        sync_df_all = pd.read_csv(args.sync_data_path)
        movie_names = np.unique(sync_df_all.movie_name.values)

        face_paths = natsorted(glob.glob(args.face_data_root + "E0030_*.csv"))
        
        session_ids = [int(re.split("[._]", path)[-2]) - 1 for path in face_paths]

        X_list = []
        Y_list = []
        subject_idx_list = []

        # NOTE: simply consider each subject has one session
        for subject_id, (session_id, face_path) in enumerate(zip(session_ids, face_paths)):
            
            sync_df = sync_df_all[sync_df_all.movie_name == movie_names[session_id]]
            
            face_df = pd.read_csv(face_path)
            # face_data = face_df.filter(like="p_", axis=1).values
            Y = face_df.drop(
                ["frame", " face_id", " timestamp", " confidence", " success"],
                axis=1,
            ).values
            Y = Y[sync_df.movie_frame.values.astype(int)]
            
            X = self.load_ecog_data(sync_df)
            
            X = torch.from_numpy(X.astype(np.float32))
            Y = torch.from_numpy(Y.astype(np.float32))

            # NOTE: deep split
            assert X.shape[0] == Y.shape[0]
            split_idx = int(X.shape[0] * args.train_ratio)
            if train:
                X = X[:split_idx]
                Y = Y[:split_idx]
            else:
                X = X[split_idx:]
                Y = Y[split_idx:]
                
            subject_id *= torch.ones(X.shape[0], dtype=torch.uint8)
            cprint(f"X: {X.shape} | Y: {Y.shape} | subject_idx: {subject_id.shape}", "cyan")

            X_list.append(X)
            Y_list.append(Y)
            subject_idx_list.append(subject_id)

            del X, Y, subject_id
            
            
    def load_ecog_data(self, sync_df: pd.DataFrame) -> np.ndarray:
        brainwave_name = sync_df.brainwave_name.values[0]
        dat = h5py.File(self.ecog_data_root + brainwave_name, "r")["ecog_dat"][()]
        
        brainwave_frames = sync_df.brainwave_frame.values.astype(int)
        ecog_data = []
        for i, frame in enumerate(tqdm(brainwave_frames)):
            if sync_df.brainwave_name.values[i] != brainwave_name:
                cprint("Loading new ECoG file.", "yellow")
                brainwave_name = sync_df.brainwave_name.values[i]
                dat = h5py.File(self.ecog_data_root + brainwave_name, "r")["ecog_dat"][()]
                
            ecog_data.append(dat[frame])
            
        return np.stack(ecog_data)
    
    
if __name__ == "__main__":
    from hydra import initialize, compose

    with initialize(version_base=None, config_path="../../configs/"):
        args = compose(config_name="ylab_ecog.yaml")

    dataset = YLabECoGDataset(args, True)