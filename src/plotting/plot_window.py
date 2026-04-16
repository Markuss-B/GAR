import matplotlib.pyplot as plt
import numpy as np

def plot_window(x, y, actvity_mapping, trainer, time=None):
    """Plot one sensor window with separate acceleration and gyroscope subplots."""
    x = np.asarray(x)
    n_steps, n_channels = x.shape
    if time is None:
        time = np.arange(n_steps) * 0.02  # 50Hz sampling rate
    else:
        time = np.asarray(time)

    channel_names = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
    acc_names = channel_names[:3]
    gyr_names = channel_names[3:6]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Window plot: activity={actvity_mapping[int(y)]}({int(y)} trainer={trainer}) ", fontsize=14)

    for i, name in enumerate(acc_names):
        axes[0].plot(time, x[:, i], label=name)
    axes[0].set_ylabel("Acceleration")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    for i, name in enumerate(gyr_names, start=3):
        axes[1].plot(time, x[:, i], label=name)
    axes[1].set_ylabel("Gyroscope")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    axes[1].set_xlabel("Seconds")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return fig, axes

def plot_window_from_df(df, activity_mapping):
    """Plot one sensor window from a DataFrame."""
    features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
    x = df[features].values
    y = df["activity"].values[0]
    trainer = df["trainer"].values[0]
    time = df["time"].values
    
    fig, axes = plot_window(x, y, activity_mapping, trainer, time)
    return fig, axes

def plot_random_windows(x, y, meta, activity_mapping, n=3, seed=123):
    """Plot n random windows from the dataset."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x), size=min(n, len(x)), replace=False)
    for idx in indices:
        plot_window(x[idx], y[idx], activity_mapping, meta[idx]['trainer'])

def plot_random_windows_of_activity(x, y, meta, activity_mapping, activity, n=3, seed=123):
    """Plot n random windows from the dataset for a specific activity."""
    # Filter windows by the selected activity
    activity_mask = y == activity
    x_filtered = x[activity_mask]
    y_filtered = y[activity_mask]
    meta_filtered = [meta for meta, keep in zip(meta, activity_mask) if keep]
    
    if len(x_filtered) == 0:
        print(f"No windows found for activity {activity} ({activity_mapping.get(activity, 'Unknown')})")
        return
    
    # Plot random windows from the filtered set
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x_filtered), size=min(n, len(x_filtered)), replace=False)
    for idx in indices:
        plot_window(x_filtered[idx], y_filtered[idx], activity_mapping, meta_filtered[idx]['trainer'])