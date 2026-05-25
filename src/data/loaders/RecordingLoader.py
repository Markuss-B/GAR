import json

import os
import pandas as pd

from process_and_save_recording import load_recording, convert_events_to_watch_time


class RecordingLoader:
    def __init__(self, session_paths, mode="windows", ignore_activities=None):
        if isinstance(session_paths, str):
            session_paths = [session_paths]

        if mode not in {"samples", "windows"}:
            raise ValueError("mode must be either 'samples' or 'windows'")

        self.session_paths = session_paths
        self.mode = mode
        self.ignore_activities = set(ignore_activities or [])

        self.ACTIVITY_MAPPING = {
            0: "Rest",
            1: "Exercise"
        }

        self.non_lifting_activities = {
            0: "Rest"
        }

    def load_data(self):
        dfs = []

        for recording_id, path in enumerate(self.session_paths):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Recording folder not found: {path}")

            accel, gyro, classification, annotation, info = load_recording(path)
            events = convert_events_to_watch_time(annotation, info)

            events = [
                e for e in events
                if e["name"] not in self.ignore_activities
            ]

            if self.mode == "samples":
                raise Exception("Loading samples not supported")
            else:
                df = self._load_windows(classification)

            df = self._add_binary_activity_labels(df, events)

            df["trainer"] = recording_id
            df["recording"] = os.path.basename(path)

            dfs.append(df)

        data_df = pd.concat(dfs, ignore_index=True)

        return data_df, self.ACTIVITY_MAPPING

    def _load_recording(path):
        accel = pd.read_csv(f"{path}/accel.csv") # timestamp, x, y, z
        gyro = pd.read_csv(f"{path}/gyro.csv") # timestamp, x, y, z
        classification = pd.read_csv(f"{path}/classification.csv") # timestamp,prediction_float,prediction_binary,start_accel_ts,end_accel_ts,start_gyro_ts,end_gyro_ts,window_data
        with open(f"{path}/annotations.json") as f:
            annotations = json.load(f)
            # annotations is a dict with keys "phoneStartTimestamp", "phoneEndTimestamp", "events": ["name", "start_time", "end_time"]

        phone_start_timestamp = annotations["phoneStartTimestamp"]
        phone_end_timestamp = annotations["phoneEndTimestamp"]

        # read watch info file info.txt which has 2 lines 
        # start_time=1776349856154
        # phone_timestamp=1776349855769
        with open(f"{path}/info.txt") as f:
            info = f.read().splitlines()
            watch_start_timestamp = int(info[0].split('=')[1])
            watch_end_timestamp = int(info[2].split('=')[1])
        
        return accel, gyro, classification, annotations, info

    def _load_windows(self, classification):
        df = classification.copy()

        if "timestamp" in df.columns:
            df["time"] = (df["timestamp"] - df["timestamp"].min()) / 1000

        return df

    def _add_binary_activity_labels(self, df, events):
        df = df.copy()

        df["activity"] = 0
        df["activity_name"] = "Rest"

        for event in events:
            mask = (
                (df["timestamp"] >= event["start_watch"]) &
                (df["timestamp"] <= event["end_watch"])
            )

            df.loc[mask, "activity"] = 1
            df.loc[mask, "activity_name"] = event["name"]

        return df

    def get_non_lifting_activities(self):
        return self.non_lifting_activities