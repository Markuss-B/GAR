import numpy as np

np.random.seed(123)

def create_windows(df, window_size=100, step=50):
    """
    Create sliding windows.

    Returns:
    - x: (num_windows, T, 6)
    - y: (num_windows,) majority label per window
    - meta: list of dicts with metadata
    """
    features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

    x, y = [], []
    meta = []

    for trainer in df["trainer"].unique():
        sub = df[df["trainer"] == trainer].copy()

        data = sub[features].values
        labels = sub["activity"].values
        times = sub["time"].values
        original_indices = sub.index.values

        N = len(sub)

        for start in range(0, N - window_size, step):
            end = start + window_size

            window = data[start:end]
            label_window = labels[start:end]
            label = np.bincount(label_window).argmax()

            x.append(window)
            y.append(label)

            meta.append({
                "trainer": trainer,
                "original_indices": original_indices[start:end],

                # important for latency
                "window_start_time": times[start],
                "window_end_time": times[end - 1],
                "window_center_time": (times[start] + times[end - 1]) / 2,

                # optional, useful for debugging
                "start_original_index": original_indices[start],
                "end_original_index": original_indices[end - 1],
            })

    return np.array(x), np.array(y), meta

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