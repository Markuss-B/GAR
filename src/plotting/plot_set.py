import numpy as np
import matplotlib.pyplot as plt

def get_activity_segments(df, trainer_id, activity_id, include_rest=False):
    """
    Returns a list of DataFrames, each one a contiguous segment ("set")
    where activity == activity_id for the given trainer_id.
    """
    d = df[df["trainer"] == trainer_id].copy()
    if not include_rest:
        d = d[d["activity"] != 0]

    # Ensure sorted in time order (prefer 'time' if present)
    if "time" in d.columns:
        d = d.sort_values("time")
    else:
        d = d.reset_index(drop=True)

    acts = d["activity"].to_numpy()

    # Identify boundaries where activity changes
    change_idx = np.where(np.diff(acts) != 0)[0] + 1
    starts = np.r_[0, change_idx]
    ends   = np.r_[change_idx, len(d)]

    segments = []
    for s, e in zip(starts, ends):
        seg = d.iloc[s:e]
        if int(seg["activity"].iloc[0]) == activity_id:
            segments.append(seg.reset_index(drop=True))

    return segments

def plot_set(df, activity_mapping, trainer_id, activity_id, set_idx=0, title_prefix="", alt_activity_title=""):
    """
    Plots one contiguous set for trainer_id + activity_id.
    """
    segments = get_activity_segments(df, trainer_id, activity_id)
    if not segments:
        raise ValueError(f"No segments found for trainer={trainer_id}, activity={activity_id}")

    if set_idx < 0 or set_idx >= len(segments):
        raise IndexError(f"set_idx out of range. Found {len(segments)} segments.")

    seg = segments[set_idx]

    # x-axis
    if "time" in seg.columns:
        x = seg["time"].to_numpy()
        x_label = "Laiks (s)"

    acc = seg[["acc_x", "acc_y", "acc_z"]].to_numpy()
    gyr = seg[["gyr_x", "gyr_y", "gyr_z"]].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].plot(x, acc[:, 0], label="acc_x")
    axes[0].plot(x, acc[:, 1], label="acc_y")
    axes[0].plot(x, acc[:, 2], label="acc_z")
    axes[0].set_ylabel("Akselerometrs")
    axes[0].legend(loc="upper right")

    axes[1].plot(x, gyr[:, 0], label="gyr_x")
    axes[1].plot(x, gyr[:, 1], label="gyr_y")
    axes[1].plot(x, gyr[:, 2], label="gyr_z")
    axes[1].set_ylabel("Žiroskops")
    axes[1].set_xlabel(x_label)
    axes[1].legend(loc="upper right")

    dur = x[-1] - x[0] if len(x) > 1 else 0.0

    activity_title = alt_activity_title if alt_activity_title else activity_mapping[activity_id]
    fig.suptitle(f"{title_prefix}Sportists {trainer_id} | Vingrinājums: {activity_title} | Pieeja #{set_idx} | ~{dur:.2f}s")

    plt.tight_layout()
    plt.show()

    return seg  # handy if you want to inspect it

def activities_for_trainer_encoded(data_df, activity_mapping, trainer_id):

    codes = (data_df.loc[data_df["trainer"] == trainer_id, "activity"]
             .dropna()
             .astype(int)
             .unique())

    pairs = [(int(c), activity_mapping[int(c)]) for c in sorted(codes)]
    return pairs