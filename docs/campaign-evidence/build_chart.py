"""Rebuild the campaign's aggregate-only, matched F1 score figure."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ROWS = {"2025-07": 190713, "2025-12": 165677}

# Frozen one-shot score files and independent review artifacts, never label/prediction data.
SOURCES = [
    (
        "Clock density weighting", "uniform component",
        "output/clock_campaign_20260923/scoring/complete_v2/result.json",
        "51ece57f5e5334c9c451e9dcab226ee1a6b404bac595492b2813f2236f33dd18",
        "output/clock_campaign_20260923/scoring/F1_RESULT.md",
        "901665bd60f18970a48476b1a8912b1ee46af75e1a410e155c2b325ead6c05de",
        "clock",
    ),
    (
        "Destination ninth peer", "temporal ninth peer",
        "private_runs/lead235_20260923/destination_fit_v1/scoring_v1/F1_score/result.json",
        "fe057ccb548c2e2f8359a7e32e5bcd1745aeddba83d345fb4fa2d61c441ac591",
        "private_runs/lead235_20260923/destination_fit_v1/scoring_v1/F1_audit/verification.json",
        "52ce50c172d4336286fa01100b2efbc21125fc15b7f72d9686a4b4079176346e",
        "destination",
    ),
    (
        "Same-stand ARR context", "airport-wide ARR context",
        "private_runs/lead235_20260923/stand_reuse_scoring_v1/F1/score_v1/result.json",
        "23f1e024a384843fb29c05e170c85c0c11b9246e205de103aff1505360abf0d7",
        "private_runs/lead235_20260923/stand_reuse_scoring_v1/F1/audit_v1/verification.json",
        "7b4ced095db87925fee830357abdcd7edcacca75058ea5b4ba7e10f1d23c2d24",
        "stand",
    ),
    (
        "Source-aware joint learning", "equal-capacity rich-only",
        "private_runs/lead235_20260924/source_aware_scoring_v1/F1_score_v1/result.json",
        "a47e051c5ab3f34d28ec2519f08ac4b2764b49d71d620e31606d62ccd393d73c",
        "review_work/lead235_20260924/source_aware_score_audit_v1/receipt_F1_v1.json",
        "2a873bf4053be5dd6761a53413caf2dfa080a7de6dd2a06e60208bbb33aae109",
        "source_aware",
    ),
    (
        "Clock-gap peer rank", "NM-proxy peer rank",
        "private_runs/lead235_20260924/peer_rank_scorer_v1/score_F1_v1/result.json",
        "00202bea95bfbe7a48f06a9bdef9ba0dadc3743e6df69084fe887d1ea081c805",
        "review_work/lead235_20260924/peer_rank_score_audit_v1/receipt_F1_v1.json",
        "85f26900eeaa5e43f43b7ca3d9718e5dc451a0fa7e16b9e637d805eb4f0d2755",
        "rank",
    ),
    (
        "ARR-shared stem", "equal-capacity detached stem",
        "private_runs/lead235_20260924/arr_aux_f1_score_v1/F1/result.json",
        "97eabf7881d5fe31048f430ec56ba52e69061fa0c835332f0ee1146c3a328634",
        "review_work/lead235_20260924/arr_aux_f1_postscore_review_v1/postscore_audit_receipt.json",
        "c4143d79c729d3b8c97ee4cbf26b807d232691974f260346d1ec79e11affb652",
        "arr_aux",
    ),
    (
        "Extra-time weather", "routine-only weather",
        "private_runs/lead235_20260924/weather_f1_score_v4/F1/result.json",
        "89ac49c14b70d9586ee645dd83552b8d3a6e9fbbe3680cfd3b166174fb937228",
        "review_work/lead235_20260924/weather_f1_postscore_audit_v1/postscore_audit_receipt.json",
        "7393000d7e226f24a8b6f36d0fc24215b39c4dbf12634ce174929fda4e7bafc4",
        "weather",
    ),
]

OFFICIAL = [
    (
        "V3", "private_runs/tail240_20260916/final_submission_v3_v2/organizer_result.json",
        "f72ed124b013230f45416ac9e26c737bd1af1948b5c172baddd9cbb734251bf3",
    ),
    (
        "V4", "output/season_strategy_20260923/blind_upload/organizer_result_v1.json",
        "59dd72d3b961791e55d53a0441bc6e3c61dc35ff4306de93d1cdf49a570f2503",
    ),
]

HEADERS = [
    "panel", "direction", "month", "comparator", "control_rmse_s",
    "candidate_rmse_s", "gain_s", "rows", "gate", "source", "source_sha256",
    "audit", "audit_sha256",
]


def verified_bytes(relative: str, expected: str) -> bytes:
    data = (ROOT / relative).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected:
        raise ValueError(f"Receipt changed: {relative} ({digest})")
    return data


def complete_comparison(data: dict, kind: str, month: str) -> tuple[float, float, float, int]:
    short = month[-2:]
    if kind == "clock":
        part = data["months"][short]["complete"]
        return part["control"]["rmse"], part["weighted"]["rmse"], part["gain_sec"], part["rows"]
    if kind == "destination":
        part = data["panels"][f"F1:{month}"]
        return part["control_rmse"], part["candidate_rmse"], part["gain_sec"], part["rows"]
    if kind == "stand":
        part = data["comparisons"][month]["control"]
        return part["control_rmse"], part["candidate_rmse"], part["gain_sec"], part["rows"]
    if kind == "source_aware":
        part = data["folds"]["F1"][month]["comparisons"]["rich_only"]["complete"]
        return part["comparator_rmse"], part["joint_rmse"], part["gain_sec"], part["rows"]
    if kind == "rank":
        part = data["months"][month]["comparisons"]["control"]["complete"]
        return part["comparator_rmse"], part["candidate_rmse"], part["gain_sec"], part["rows"]
    if kind == "arr_aux":
        part = data["months"][month]["comparisons"]["detached"]
    elif kind == "weather":
        part = data["months"][month]["comparisons"]["routine"]
    else:
        raise ValueError(f"Unknown source kind: {kind}")
    return part["comparator_rmse"], part["candidate_rmse"], part["gain_sec"], part["rows"]


def aggregate_rows() -> list[dict]:
    result = []
    for name, comparator, src, src_hash, audit, audit_hash, kind in SOURCES:
        data = json.loads(verified_bytes(src, src_hash))
        audited = verified_bytes(audit, audit_hash)
        if kind == "clock":
            assert src_hash.encode() in audited and data["advancement"]["passed"] is False
        else:
            review = json.loads(audited)
            audit_result_hash = review.get("score_sha256", review.get("f1_result_sha256", review.get("score_result_sha256", review.get("result_sha256"))))
            assert audit_result_hash == src_hash, (kind, "independent audit did not bind score")
            gate = data["f1_gate"] if kind in ("source_aware", "rank", "arr_aux", "weather") else data["gate"]
            assert gate["passed"] is False, (kind, "gate changed")
            assert review.get("f1_gate_passed", review.get("gate_passed", False)) is False
        for month in ROWS:
            control, candidate, gain, count = complete_comparison(data, kind, month)
            assert count == ROWS[month], (kind, month, count)
            assert abs((control - candidate) - gain) < 1e-7, (kind, month, "gain arithmetic")
            result.append(dict(panel="matched_local_F1", direction=name, month=month,
                               comparator=comparator, control_rmse_s=f"{control:.9f}",
                               candidate_rmse_s=f"{candidate:.9f}", gain_s=f"{gain:.9f}",
                               rows=count, gate="F1_STOP", source=src, source_sha256=src_hash,
                               audit=audit, audit_sha256=audit_hash))
    for name, src, src_hash in OFFICIAL:
        data = json.loads(verified_bytes(src, src_hash))
        assert data["status"] == "Succeeded" and data["used_pairs"] == 344841
        assert data["score"] > 0
        result.append(dict(panel="official_2026", direction=name, month="2026-01+07",
                           comparator="", control_rmse_s="", candidate_rmse_s=f"{data['score']:.4f}",
                           gain_s="", rows=data["used_pairs"], gate="Succeeded", source=src,
                           source_sha256=src_hash, audit="", audit_sha256=""))
    return result


def draw(rows: list[dict]) -> None:
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(12.6, 8.9), constrained_layout=False)
    outer = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[6.5, 2.3],
                             width_ratios=[1, 1], left=.30, right=.965, top=.84,
                             bottom=.088, hspace=.65, wspace=.18)
    fig.text(.045, .963, "Campaign score evidence", fontsize=19, weight="bold", color="#172d39")
    fig.text(.045, .927, "Every matched F1 direction stopped before F3 or release. Positive peer gain is not qualification.",
             fontsize=10.1, color="#53626b")
    names = [source[0] for source in SOURCES]
    panel = [row for row in rows if row["panel"] == "matched_local_F1"]
    colors = {"2025-07": "#176e78", "2025-12": "#bd663c"}
    bounds = {"2025-07": (-1.35, 3.8), "2025-12": (-1.35, 1.6)}
    for col, month in enumerate(ROWS):
        ax = fig.add_subplot(outer[0, col])
        ax.axvline(0, color="#77838b", linewidth=1.0, zorder=0)
        for i, name in enumerate(names):
            row = next(row for row in panel if row["direction"] == name and row["month"] == month)
            gain = float(row["gain_s"])
            y = len(names) - i - 1
            ax.barh(y, gain, height=.52, color=colors[month], alpha=.88, zorder=2)
            offset = .10 if gain >= 0 else -.10
            ax.text(gain + offset, y, f"{gain:+.3f}", ha="left" if gain >= 0 else "right",
                    va="center", fontsize=9, color="#172d39")
            if col == 0:
                ax.text(-.06, y, name, transform=ax.get_yaxis_transform(), ha="right", va="center",
                        fontsize=9.3, color="#172d39")
        ax.set_title(f"{'July' if col == 0 else 'December'} 2025", loc="left",
                     fontsize=11, weight="bold", color="#172d39", pad=10)
        ax.set(xlim=bounds[month], ylim=(-.6, len(names) - .4), yticks=[])
        ax.set_xlabel("Complete-cohort RMSE gain (seconds)", labelpad=5)
        ax.xaxis.set_major_locator(MultipleLocator(1 if col == 0 else .5))
        ax.grid(axis="x", color="#dde3e7", linewidth=.7, zorder=0)
        ax.tick_params(axis="x", colors="#53626b")
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#bdc8ce")
    ax = fig.add_subplot(outer[1, :])
    official = [row for row in rows if row["panel"] == "official_2026"]
    ax.set_title("Official ranking  |  January + July 2026 (separate absolute-score scale)", loc="left",
                 fontsize=11, weight="bold", color="#172d39", pad=9)
    for i, row in enumerate(official):
        score = float(row["candidate_rmse_s"])
        y = len(official) - i - 1
        ax.scatter(score, y, s=85, color="#176e78" if row["direction"] == "V3" else "#bd663c", zorder=3)
        ax.text(score + .4, y, f"{score:.4f} s", va="center", color="#172d39", fontsize=10, weight="bold")
        ax.text(-.06, y, row["direction"], transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=10, weight="bold", color="#172d39")
    ax.set(xlim=(276.5, 285.0), ylim=(-.5, 1.5), yticks=[], xlabel="Official RMSE (seconds; lower is better)")
    ax.grid(axis="x", color="#dde3e7", linewidth=.7)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#bdc8ce")
    fig.text(.045, .024, "Local bars: same-origin control minus candidate. All 7 F1 gates failed."
             "  |  Source SHA-256s and comparator names: aggregate_scores.csv", fontsize=8.3, color="#53626b")
    fig.savefig(OUT / "campaign_scores.png", dpi=155)
    plt.close(fig)


def main() -> None:
    rows = aggregate_rows()
    with (OUT / "aggregate_scores.csv").open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    draw(rows)
    print(f"Verified {len(SOURCES)} matched F1 directions and {len(OFFICIAL)} official receipts; wrote {len(rows)} aggregate rows.")


if __name__ == "__main__":
    main()
