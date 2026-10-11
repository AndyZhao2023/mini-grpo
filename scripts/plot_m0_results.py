"""Plot recorded M0 internal-validation outcomes from the run summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(fresh_summary: Path, mu2_summary: Path, output: Path) -> None:
    fresh = json.loads(fresh_summary.read_text())
    mu2 = json.loads(mu2_summary.read_text())
    for run in (fresh, mu2):
        if run["status"] != "completed" or any(
            run[stage]["question_count"] != 200 for stage in ("baseline", "final")
        ):
            raise ValueError("chart requires completed runs on the fixed 200-question validation set")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
    labels = ["Fresh rollout × 2", "Same rollout μ=2"]
    runs = [fresh, mu2]
    initial = [run["baseline"]["correct_count"] for run in runs]
    final = [run["final"]["correct_count"] for run in runs]
    xs = [0, 1]
    bars1 = ax1.bar([x - 0.18 for x in xs], initial, width=0.34,
                    label="Initial", color="#8b9cac")
    bars2 = ax1.bar([x + 0.18 for x in xs], final, width=0.34,
                    label="After two updates", color="#246b8e")
    ax1.bar_label(bars1, labels=[f"{n}/200" for n in initial], padding=3)
    ax1.bar_label(bars2, labels=[f"{n}/200" for n in final], padding=3)
    ax1.set_xticks(xs, labels)
    ax1.set_ylim(0, 12)
    ax1.set_ylabel("Strictly correct questions")
    ax1.set_title("Fixed internal validation")
    ax1.legend(frameon=False, loc="upper right")

    reason_specs = [
        ("correct_answer", "Correct", "#2c8c76"),
        ("incorrect_answer", "Incorrect final line", "#bd6b4d"),
        ("invalid_final_answer", "Invalid final line", "#a6a6a6"),
        ("missing_final_answer", "Missing final line", "#d6ac52"),
    ]
    left = [0, 0]
    for key, label, color in reason_specs:
        values = [fresh[stage]["reason_counts"].get(key, 0) for stage in ("baseline", "final")]
        bars = ax2.barh([0, 1], values, left=left, height=0.48, label=label, color=color)
        for bar, value in zip(bars, values):
            if value >= 8:
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_y() + bar.get_height() / 2, str(value),
                         ha="center", va="center", color="white" if key != "missing_final_answer" else "#2a2a2a")
        left = [old + value for old, value in zip(left, values)]
    ax2.set_yticks([0, 1], ["Initial", "After two updates"])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 200)
    ax2.set_xlabel("Questions (out of 200)")
    ax2.set_title("Fresh-rollout verifier outcomes")
    ax2.legend(frameon=False, bbox_to_anchor=(0, -0.22), loc="upper left", ncol=2)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-summary", type=Path,
                        default=Path("results/m0/end-to-end-v2/summary.json"))
    parser.add_argument("--mu2-summary", type=Path,
                        default=Path("results/m0/end-to-end-mu2-v1/summary.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("docs/assets/m0-validation-results.png"))
    args = parser.parse_args()
    main(args.fresh_summary, args.mu2_summary, args.output)
