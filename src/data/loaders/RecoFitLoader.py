import os
from scipy.io import loadmat
import numpy as np
import pandas as pd

class RecoFitLoader:
    def __init__(self, mat_dir, transform_units=False):
        self.mat_dir = mat_dir
        self.transform_units = transform_units

    def load_data(self):
        if not os.path.exists(self.mat_dir):
            raise FileNotFoundError(f"Directory not found: {self.mat_dir}")
        
        files = [f for f in os.listdir(self.mat_dir) if f.endswith(".mat")]
        if not files:
            raise ValueError(f"No .mat files found in {self.mat_dir}")
        
        dfs = []
        for i, f in enumerate(files):
            datamat = loadmat(os.path.join(self.mat_dir, f))
            
            # Extract metadata
            recordingID = datamat["recordingID"][0][0]
            subjectID = datamat["subjectID"][0][0]
            
            accel = datamat["accel"]
            gyro = datamat["gyro"]
            
            if self.transform_units:
                accel[:, 1:] *= 9.81  # Convert accel from g to m/s^2
                gyro[:, 1:] *= np.pi / 180  # Convert gyro from deg/s to rad/s
            
            act_start = datamat["act_start"].reshape(-1)
            act_end = datamat["act_end"].reshape(-1)
            act_name = datamat["act_name"].reshape(-1)
            act_name = np.array([x[0] for x in act_name])
            
            time = accel[:, 0]
            activity = np.full(time.shape, None, dtype=object)
            for s, e, name in zip(act_start, act_end, act_name):
                mask = (time >= s) & (time <= e)
                activity[mask] = name
            
            activity = pd.Series(activity).fillna("Unknown").to_numpy()
            
            dfi = pd.DataFrame({
                "acc_x": accel[:, 1],
                "acc_y": accel[:, 2],
                "acc_z": accel[:, 3],
                "gyr_x": gyro[:, 1],
                "gyr_y": gyro[:, 2],
                "gyr_z": gyro[:, 3],
                "activity": activity,
                "trainer": i,
                # "subjectID": subjectID,  # Uncomment if needed
                # "recordingID": recordingID,  # Uncomment if needed
                "time": time,
            })
            dfs.append(dfi)
        
        data_df = pd.concat(dfs, ignore_index=True)
        
        # Build activity index and mapping
        activity_index = {act: i for i, act in enumerate(sorted(data_df["activity"].unique()))}
        data_df["activity"] = data_df["activity"].map(activity_index).astype("int32")
        ACTIVITY_MAPPING = {v: str(k) for k, v in activity_index.items()}
        
        return data_df, ACTIVITY_MAPPING