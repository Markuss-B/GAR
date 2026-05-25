from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd

from edit_recording import load_recording
from process_and_save_recording import convert_events_to_watch_time

TITLE_SIZE = 13
LABEL_SIZE = 14
TICK_SIZE = 11
LEGEND_SIZE = 14

def apply_accumulator_aggregation(df, threshold=3):
    preds = df["prediction_binary"].values
    aggregated = []

    state = 0
    accumulator = 0

    for pred in preds:
        if pred == 1:
            accumulator += 1
        else:
            accumulator -= 1

        accumulator = max(0, min(threshold, accumulator))

        if state == 0 and accumulator >= threshold:
            state = 1
        elif state == 1 and accumulator <= 0:
            state = 0

        aggregated.append(state)

    return aggregated


def to_relative_seconds(classification, events):
    start_time = classification["timestamp"].min()

    df = classification.copy()
    df["time_s"] = (df["timestamp"] - start_time) / 1000

    converted_events = []
    for event in events:
        converted_events.append({
            **event,
            "start_s": (event["start_watch"] - start_time) / 1000,
            "end_s": (event["end_watch"] - start_time) / 1000,
        })

    return df, converted_events

def to_relative_minutes(classification, events):
    start_time = classification["timestamp"].min()

    df = classification.copy()
    df["time_min"] = (df["timestamp"] - start_time) / 60000

    converted_events = []
    for event in events:
        converted_events.append({
            **event,
            "start_min": (event["start_watch"] - start_time) / 60000,
            "end_min": (event["end_watch"] - start_time) / 60000,
        })

    return df, converted_events


def plot_signal_on_axis(ax, classification, events, title="", event_color_map=None):
    df, rel_events = to_relative_minutes(classification, events)

    for event in rel_events:
        ax.axvspan(
            event["start_min"],
            event["end_min"],
            color=event_color_map[event["name"]],
            alpha=0.3,
            linewidth=0
        )

    ax.step(
        df["time_min"],
        df["prediction_float"],
        where="post",
        color="tab:red",
        alpha=0.25,
        linewidth=1
    )

    ax.step(
        df["time_min"],
        df["prediction_binary"],
        where="post",
        color="black",
        linewidth=0.9
    )

    ax.axhline(
        y=0.5,
        color="tab:red",
        linestyle="--",
        linewidth=0.8,
        alpha=0.15
    )

    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.set_ylim(-0.1, 1.05)
    ax.set_yticks([0, 1])
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_multiple(strategies, events):
    fig, axs = plt.subplots(
        len(strategies),
        1,
        figsize=(18, 6),
        sharex=True,
        sharey=True
    )

    unique_events = sorted({event["name"] for event in events})
    colors = plt.get_cmap("tab20", len(unique_events))

    event_color_map = {
        name: colors(i)
        for i, name in enumerate(unique_events)
    }

    for ax, (name, df) in zip(axs, strategies):
        plot_signal_on_axis(
            ax,
            df,
            events,
            title=name,
            event_color_map=event_color_map
        )

    legend_items = [
        Patch(facecolor="gray", alpha=0.3, label="Vingrinājums"),
        Line2D([0], [0], color="tab:red", alpha=0.35, label="Varbūtība"),
        Line2D([0], [0], color="black", label="Stāvoklis"),
        Line2D([0], [0], color="tab:red", linestyle="--", alpha=0.6, label="Slieksnis 0,5"),
    ]

    fig.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.2, 1.02),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_SIZE
    )

    axs[-1].set_xlabel("Laiks (min)", fontsize=LABEL_SIZE)

    origin_ms = strategies[0][1]["timestamp"].min()
    end_ms = strategies[0][1]["timestamp"].max()

    for ax in axs:
        ax.set_xlim(-0.6, ((end_ms - origin_ms) / 60000) + 0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def main():
    path = "session_1777104300830"
    session_id = 5

    accel, gyro, classification, annotation, info = load_recording(path)

    events_watch_time = convert_events_to_watch_time(annotation, info)

    ignore_list = [
        "Treadmill Running",
        "Side Crunches",
    ]

    filtered_events = [
        e for e in events_watch_time
        if e["name"] not in ignore_list
    ]

    raw_df = classification.copy()

    raw_acc3_df = raw_df.copy()
    raw_acc3_df["prediction_binary"] = apply_accumulator_aggregation(
        raw_acc3_df,
        threshold=3
    )

    preds = pd.read_csv("prepared_sessions/fine_tuned_test_predictions.csv")
    preds = preds[preds["session_id"] == session_id].reset_index(drop=True)

    if len(preds) != len(classification):
        raise ValueError(
            f"Prediction count mismatch: preds={len(preds)}, classification={len(classification)}"
        )

    finetuned_df = classification.copy()
    finetuned_df["prediction_float"] = preds["prediction_finetuned_float"].values
    finetuned_df["prediction_binary"] = preds["prediction_finetuned_binary"].values

    finetuned_acc3_df = finetuned_df.copy()
    finetuned_acc3_df["prediction_binary"] = apply_accumulator_aggregation(
        finetuned_acc3_df,
        threshold=3
    )

    strategies = [
        ("Sākotnējais", raw_df),
        ("Sākotnējais + akumulators 3", raw_acc3_df),
        ("Pielāgotais", finetuned_df),
        ("Pielāgotais + akumulators 3", finetuned_acc3_df),
    ]

    plot_multiple(strategies, filtered_events)


if __name__ == "__main__":
    main()