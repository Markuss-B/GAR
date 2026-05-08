import os
from scipy.io import loadmat
import pandas as pd
import numpy as np

class MyoGymLoader:
    def __init__(self, mat_path, transform_units=False):
        self.mat_path = mat_path
        self.transform_units = transform_units
        self.ACTIVITY_MAPPING = {
            0: "No activity identified",
            1: "Seated Cable Rows",
            2: "One-Arm Dumbbell Row",
            3: "Wide-Grip Pulldown Behind The Neck",
            4: "Bent Over Barbell Row",
            5: "Reverse Grip Bent-Over Row",
            6: "Wide-Grip Front Pulldown",
            7: "Bench Press",
            8: "Incline Dumbbell Flyes",
            9: "Incline Dumbbell Press",
            10: "Dumbbell Flyes",
            11: "Pushups",
            12: "Leverage Chest Press",
            13: "Close-Grip Barbell Bench Press",
            14: "Bar Skullcrusher",
            15: "Triceps Pushdown",
            16: "Bench Dip / Dip",
            17: "Overhead Triceps Extension",
            18: "Tricep Dumbbell Kickback",
            19: "Spider Curl",
            20: "Dumbbell Alternate Bicep Curl",
            21: "Incline Hammer Curl",
            22: "Concentration Curl",
            23: "Cable Curl",
            24: "Hammer Curl",
            25: "Upright Barbell Row",
            26: "Side Lateral Raise",
            27: "Front Dumbbell Raise",
            28: "Seated Dumbbell Shoulder Press",
            29: "Car Drivers",
            30: "Lying Rear Delt Raise"
        }
        self.non_lifting_activities = {0: "No activity identified"}  # Only "No activity identified" is non-lifting

    def load_data(self):
        if not os.path.exists(self.mat_path):
            raise FileNotFoundError(f"File not found: {self.mat_path}")
        
        # Load the MyoGym data
        datamat = loadmat(self.mat_path)
        raw_data = pd.DataFrame(datamat["raw_data"])
        label_data = pd.DataFrame(datamat["raw_data_labels"])
        
        # Extract and rename accelerometer and gyroscope columns
        raw_data.rename(columns={
            9: "time_acc", 
            10: "acc_x",
            11: "acc_y",
            12: "acc_z", 
            13: "time_gyr", 
            14: "gyr_x",
            15: "gyr_y",
            16: "gyr_z"
        }, inplace=True)
        raw_data = raw_data[["time_acc", "acc_x", "acc_y", "acc_z", "time_gyr", "gyr_x", "gyr_y", "gyr_z"]]

        if self.transform_units:
            raw_data[["acc_x", "acc_y", "acc_z"]] *= 9.81  # Convert accel from g to m/s^2
            raw_data[["gyr_x", "gyr_y", "gyr_z"]] *= np.pi / 180  # Convert gyro from deg/s to rad/s
        
        # Rename label columns
        label_data.rename(columns={0: "activity", 1: "trainer"}, inplace=True)
        
        # Map activity labels
        label_data["activity"] = label_data["activity"].replace({99: 0})
        
        # Concatenate raw data and labels
        data_df = pd.concat([raw_data, label_data], axis=1)

        # Sort and remove duplicates
        data_df = data_df.sort_values(by=['trainer', 'time_acc'], ascending=True)
        data_df = data_df.drop_duplicates()

        # # Check how many hours are lost to duplicates
        # raw_count = len(datamat["raw_data"])
        # clean_count = len(data_df)
        # lost_seconds = (raw_count - clean_count) / 50

        # print(f"Hours lost to duplicates/cleaning: {lost_seconds/3600:.2f}")
        
        # Create synthetic timestamp and remove sensor times
        fq = 50  # Assumed frequency from notebook
        data_df["time"] = data_df.groupby("trainer").cumcount()
        data_df["time"] /= fq
        data_df.drop(columns=["time_acc", "time_gyr"], inplace=True)
        
        return data_df, self.ACTIVITY_MAPPING
    
    def get_non_lifting_activities(self):
        return self.non_lifting_activities
