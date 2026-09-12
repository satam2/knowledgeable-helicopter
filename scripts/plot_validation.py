"""Export measured fold and cohort evidence as a static research figure."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from taxiout.artifacts import read_json
from taxiout.config import ROOT


def main():
    comparison = read_json(ROOT / "reports/comparison.json")["candidates"]
    release = read_json(ROOT / "reports/release_manifest.json")
    eligible = [name for name, result in comparison.items() if "season_score" in result]
    best = sorted(eligible, key=lambda name: comparison[name]["season_score"])[:3]
    names = list(dict.fromkeys(["direct", *best, release["candidate"]]))
    names = [name for name in names if name in eligible]
    folds = ["F1", "F2", "F3"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    colors = ["#2864a5", "#23836d", "#b74b59", "#6b617e", "#626c72"]
    width = .8 / len(names)
    for index, name in enumerate(names):
        positions = np.arange(3) - .4 + width / 2 + index * width
        values = [comparison[name]["folds"][fold]["rmse_sec"] for fold in folds]
        axes[0].bar(positions, values, width, label=name.replace("_", " "), color=colors[index])
    axes[0].set_xticks(np.arange(3), ["July (F1)", "October (F2)", "November (F3)"])
    axes[0].set_ylabel("RMSE (seconds)")
    axes[0].set_title("Complete development cohorts")
    axes[0].legend(frameon=False, fontsize=8)
    selected = comparison[release["candidate"]]
    values = [selected["missing_proxy"][fold]["sse_share"] * 100 for fold in folds]
    values.append(release["confirmation"]["metrics"]["slices"]["proxy_status"]["missing"]["sse_share"] * 100)
    axes[1].barh(["July (F1)", "October (F2)", "November (F3)", "December (C1)"], values, color="#23836d")
    axes[1].set_xlim(0, 100)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Share of total squared error (%)")
    axes[1].set_title("Selected model: missing NM proxy")
    for index, value in enumerate(values):
        axes[1].text(min(value + 1, 92), index, f"{value:.1f}%", va="center", fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
    axes[0].grid(axis="y", alpha=.2)
    axes[1].grid(axis="x", alpha=.2)
    output = ROOT / "reports/figures"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "validation.png", dpi=180)
    fig.savefig(output / "validation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
