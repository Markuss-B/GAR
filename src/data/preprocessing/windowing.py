import numpy as np

np.random.seed(123)

def create_windows(df, window_size = 100, step = 50):
    """
    Create sliding windows.

    Parameters:
    - df: pandas DataFrame
    - window_size: number of timesteps per window
    - step: stride between windows

    Returns:
    - x: (num_windows, T, 6)
    - y: (num_windows,) majority label per window
    - trainer_ids: (num_windows,) trainer per window
    - timestamps: (num_windows, T) timestamps per window
    """
    features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

    x, y, trainer_ids, timestamps = [], [], [], []

    for trainer in df["trainer"].unique():
        sub = df[df["trainer"] == trainer]

        data = sub[features].values
        labels = sub["activity"].values
        times = sub["time"].values

        N = len(sub)

        for start in range(0, N - window_size, step):
            end = start + window_size

            window = data[start:end]
            label_window = labels[start:end]
            label = np.bincount(label_window).argmax()
            time_window = times[start:end]

            x.append(window)
            y.append(label)
            trainer_ids.append(trainer)
            timestamps.append(time_window)
    
    return np.array(x), np.array(y), np.array(trainer_ids), np.array(timestamps)

def get_random_windows(x, y, t, timestamps, activity_mapping=None, activity=None, n=3, seed=123):
    """Get n random windows from the dataset, optionally filtered by activity."""
    rng = np.random.default_rng(seed)

    if activity is not None:
        mask = y == activity
        x = x[mask]
        y = y[mask]
        t = t[mask]
        timestamps = timestamps[mask]

    if len(x) == 0:
        print(f"No windows found for activity {activity} ({activity_mapping.get(activity, 'Unknown')})")
        return [], [], [], []

    indices = rng.choice(len(x), size=min(n, len(x)), replace=False)
    return x[indices], y[indices], t[indices], timestamps[indices]