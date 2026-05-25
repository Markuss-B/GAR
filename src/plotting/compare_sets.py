import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from plotting import get_activity_segments


def compare_sets(
    df_left,
    activity_mapping_left,
    trainer_left,
    activity_id_left,
    df_right,
    activity_mapping_right,
    trainer_right,
    activity_id_right,
    set_idx_left=0,
    set_idx_right=0,
    left_label="Left dataset",
    right_label="Right dataset",
    title_prefix="",
):
    """Compare one contiguous activity set from two datasets."""
    left_segments = get_activity_segments(df_left, trainer_left, activity_id_left)
    if not left_segments:
        raise ValueError(
            f"No segments found for left dataset: trainer={trainer_left}, activity={activity_id_left}"
        )
    right_segments = get_activity_segments(df_right, trainer_right, activity_id_right)
    if not right_segments:
        raise ValueError(
            f"No segments found for right dataset: trainer={trainer_right}, activity={activity_id_right}"
        )

    if set_idx_left < 0 or set_idx_left >= len(left_segments):
        raise IndexError(
            f"set_idx_left out of range. Found {len(left_segments)} segments."
        )
    if set_idx_right < 0 or set_idx_right >= len(right_segments):
        raise IndexError(
            f"set_idx_right out of range. Found {len(right_segments)} segments."
        )

    left_seg = left_segments[set_idx_left]
    right_seg = right_segments[set_idx_right]

    if "time" in left_seg.columns:
        x_left = left_seg["time"].to_numpy()
    else:
        x_left = np.arange(len(left_seg), dtype=float)

    if "time" in right_seg.columns:
        x_right = right_seg["time"].to_numpy()
    else:
        x_right = np.arange(len(right_seg), dtype=float)

    x_label = "Time (s)" if "time" in left_seg.columns or "time" in right_seg.columns else "Index"

    acc_left = left_seg[["acc_x", "acc_y", "acc_z"]].to_numpy()
    gyr_left = left_seg[["gyr_x", "gyr_y", "gyr_z"]].to_numpy()
    acc_right = right_seg[["acc_x", "acc_y", "acc_z"]].to_numpy()
    gyr_right = right_seg[["gyr_x", "gyr_y", "gyr_z"]].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex='col', sharey='row')

    axes[0, 0].plot(x_left, acc_left[:, 0], label="acc_x", linewidth=1.2)
    axes[0, 0].plot(x_left, acc_left[:, 1], label="acc_y", linewidth=1.2)
    axes[0, 0].plot(x_left, acc_left[:, 2], label="acc_z", linewidth=1.2)
    axes[0, 0].set_ylabel("Acceleration")
    axes[0, 0].set_title(f"{left_label} — Acceleration")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    axes[1, 0].plot(x_left, gyr_left[:, 0], label="gyr_x", linewidth=1.2)
    axes[1, 0].plot(x_left, gyr_left[:, 1], label="gyr_y", linewidth=1.2)
    axes[1, 0].plot(x_left, gyr_left[:, 2], label="gyr_z", linewidth=1.2)
    axes[1, 0].set_ylabel("Gyroscope")
    axes[1, 0].set_xlabel(x_label)
    axes[1, 0].set_title(f"{left_label} — Gyroscope")
    axes[1, 0].legend(loc="upper right", fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    axes[0, 1].plot(x_right, acc_right[:, 0], label="acc_x", linewidth=1.2)
    axes[0, 1].plot(x_right, acc_right[:, 1], label="acc_y", linewidth=1.2)
    axes[0, 1].plot(x_right, acc_right[:, 2], label="acc_z", linewidth=1.2)
    axes[0, 1].set_title(f"{right_label} — Acceleration")
    axes[0, 1].legend(loc="upper right", fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    axes[1, 1].plot(x_right, gyr_right[:, 0], label="gyr_x", linewidth=1.2)
    axes[1, 1].plot(x_right, gyr_right[:, 1], label="gyr_y", linewidth=1.2)
    axes[1, 1].plot(x_right, gyr_right[:, 2], label="gyr_z", linewidth=1.2)
    axes[1, 1].set_xlabel(x_label)
    axes[1, 1].set_title(f"{right_label} — Gyroscope")
    axes[1, 1].legend(loc="upper right", fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    left_activity = activity_mapping_left.get(activity_id_left, activity_id_left)
    right_activity = activity_mapping_right.get(activity_id_right, activity_id_right)
    fig.suptitle(
        f"{title_prefix} \n"
        f"{left_label}: trainer {trainer_left} — {left_activity} | set {set_idx_left}    "
        f"{right_label}: trainer {trainer_right} — {right_activity} | set {set_idx_right}",
        fontsize=14,
        fontweight='bold',
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return left_seg, right_seg


def main():
    parser = argparse.ArgumentParser(
        description="Compare one exercise set from MyoGym and RecoFit datasets."
    )
    root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--myogym-path",
        default=str(root / "data" / "datasets" / "MyoGym.mat"),
        help="Path to MyoGym.mat",
    )
    parser.add_argument(
        "--recofit-path",
        default=str(root / "data" / "datasets" / "RecoFit"),
        help="Path to RecoFit dataset directory",
    )
    parser.add_argument("--trainer-left", type=int, default=7, help="Trainer ID for MyoGym")
    parser.add_argument("--activity-left", type=int, default=11, help="Activity ID for MyoGym")
    parser.add_argument("--trainer-right", type=int, default=25, help="Trainer ID for RecoFit")
    parser.add_argument("--activity-right", type=int, default=39, help="Activity ID for RecoFit")
    parser.add_argument("--set-idx-left", type=int, default=0, help="Set index for MyoGym")
    parser.add_argument("--set-idx-right", type=int, default=0, help="Set index for RecoFit")
    parser.add_argument("--left-label", default="MyoGym", help="Label for left dataset")
    parser.add_argument("--right-label", default="RecoFit", help="Label for right dataset")
    parser.add_argument("--title-prefix", default="Exercise comparison", help="Title prefix for the comparison figure")
    args = parser.parse_args()

    from data.loaders import MyoGymLoader, RecoFitLoader

    myogym_loader = MyoGymLoader(args.myogym_path, transform_units=True)
    recofit_loader = RecoFitLoader(args.recofit_path, transform_units=True)
    df_left, mapping_left = myogym_loader.load_data()
    df_right, mapping_right = recofit_loader.load_data()

    compare_sets(
        df_left,
        mapping_left,
        trainer_left=args.trainer_left,
        activity_id_left=args.activity_left,
        df_right=df_right,
        activity_mapping_right=mapping_right,
        trainer_right=args.trainer_right,
        activity_id_right=args.activity_right,
        set_idx_left=args.set_idx_left,
        set_idx_right=args.set_idx_right,
        left_label=args.left_label,
        right_label=args.right_label,
        title_prefix=args.title_prefix,
    )


if __name__ == "__main__":
    main()
