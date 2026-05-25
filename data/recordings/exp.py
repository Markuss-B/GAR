import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_sliding_window_interpretation():
    # Parameters
    sampling_rate = 50
    window_length = 2.0
    step = 1.0

    exercise_start = 0.6
    exercise_end = 3.7

    window_starts = [0, 1, 2, 3]
    window_ends = [s + window_length for s in window_starts]

    fig, ax = plt.subplots(figsize=(10, 4))

    # Annotated exercise interval
    ax.add_patch(
        Rectangle(
            (exercise_start, 2.6),
            exercise_end - exercise_start,
            0.25,
            alpha=0.25
        )
    )
    ax.text(
        (exercise_start + exercise_end) / 2,
        2.95,
        "Anotētais vingrinājuma segments",
        ha="center",
        va="bottom",
        fontsize=11
    )

    # Exercise start/end markers
    ax.axvline(exercise_start, linestyle="--", linewidth=1.3)
    ax.axvline(exercise_end, linestyle="--", linewidth=1.3)

    ax.text(exercise_start, 3.15, "sākums", ha="center", fontsize=10)
    ax.text(exercise_end, 3.15, "beigas", ha="center", fontsize=10)

    # Sliding windows
    y_positions = [1.8, 1.4, 1.0, 0.6]

    for i, (start, end, y) in enumerate(zip(window_starts, window_ends, y_positions), start=1):
        ax.add_patch(
            Rectangle(
                (start, y),
                window_length,
                0.25,
                fill=False,
                linewidth=2
            )
        )

        ax.text(
            start + (window_length / 2) + 0.25,
            y + 0.33,
            f"Logs {i}",
            ha="center",
            fontsize=10
        )

        # Prediction becomes available only at the end of the window
        ax.plot(end, y+0.12, "o", markersize=5)
        ax.vlines(end, y, 0.25, linestyle=":", linewidth=1)

        ax.text(
            end,
            0.05,
            "klasifikācija\npieejama",
            ha="center",
            va="top",
            fontsize=8
        )

    # Real-time interpretation
    ax.annotate(
        "Reāllaikā signāls kļūst pieejams\nloga beigu brīdī",
        xy=(3.1, 1.5),
        xytext=(3.4, 1.7),
        arrowprops=dict(arrowstyle="->", linewidth=1.5),
        fontsize=10
    )

    # Segmentation interpretation
    ax.annotate(
        "Segmentēšanā pēc noteikšanas\nvar atsaukties uz loga sākumu",
        xy=(exercise_start, 1.8),
        xytext=(0.1, 0.6),
        arrowprops=dict(arrowstyle="->", linewidth=1.5),
        fontsize=10
    )

    # Formatting
    ax.set_xlim(0, 6)
    ax.set_ylim(-0.25, 3.4)
    ax.set_xlabel("Laiks (s)", fontsize=12)
    ax.set_yticks([])
    ax.set_title("Slīdošā loga interpretācija reāllaika sistēmā", fontsize=14)

    ax.grid(axis="x", alpha=0.25)

    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.show()

plot_sliding_window_interpretation()