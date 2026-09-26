"""Rebuild the campaign's aggregate-only, matched F1 score figure."""

from __future__ import annotations

import csv
import hashlib
import json
import math
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
    (
        "Route-duration centering", "byte-identical raw/clean",
        "private_runs/lead235_20260925/route_duration_f1_score_v2/F1/result.json",
        "ce4176f4d86010d4cfbc3e10c0723d38304ab701ab85592bd5c93fc9cbbfed30",
        "review_work/lead235_20260925/route_duration_f1_postscore_audit_v1/POSTSCORE_AUDIT.json",
        "942ea22b606d24b1a06bb0550d6c540bc6ea681209064795b5dab41bdcf07c07",
        "route_duration",
    ),
    (
        "Scheduled inbound density", "same-peer landed density",
        "private_runs/lead235_20260925/scheduled_f1_score_v5/F1/result.json",
        "b70208d48cc4e29e8e838affaabdfa876ba7b701d610490ea1d13b909eeb8e52",
        "private_runs/lead235_20260925/scheduled_f1_score_v5/F1/POSTSCORE_AUDIT.json",
        "33f1a78c6a261fa0cfd271a4f744213f2808b6707e87441ae8f06a751f01e764",
        "scheduled",
    ),
    (
        "Current-expert context gate", "May-fitted intercept gate",
        "private_runs/lead235_20260925/current_gate_score_v1/F1/result.json",
        "de5cb8142f5b0270b2101687d61b12609d95cbd3b942a30d6a9d999d8f47ac65",
        "private_runs/lead235_20260925/current_gate_score_v1/F1/POSTSCORE_AUDIT.json",
        "6431af709db34ad7b525306969571d58947711bade3ff08afe8ace8ff8b3f2bd",
        "current_gate",
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

ARR_DIAGNOSTIC = (
    "review_work/lead235_20260925/arr_residual_fit_v1/SCORE_REPORT.json",
    "d3ad55d4803d26d7c6ab932580522665663869d0215c4780123c47ff916efb87",
    "review_work/lead235_20260925/arr_residual_postscore_review_v1/POSTSCORE_RECEIPT.json",
    "704100a02eb04d3b5d1b2021ebad000dfc09859515a423c6fd0adff5e86202a9",
)
ARR_ROWS = {"october": ("2025-10", 185674), "november": ("2025-11", 162332)}

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
    if kind == "scheduled":
        part = data["months"][month]["comparisons"]["landed"]
        return part["comparator_rmse"], part["candidate_rmse"], part["gain_sec"], part["rows"]
    if kind == "current_gate":
        part = data["months"][month]["comparisons"]["intercept"]
        return part["comparator_rmse"], part["candidate_rmse"], part["gain_sec"], part["rows"]
    if kind == "arr_aux":
        part = data["months"][month]["comparisons"]["detached"]
    elif kind == "weather":
        part = data["months"][month]["comparisons"]["routine"]
    elif kind == "route_duration":
        part = data["months"][month]["comparisons"]["raw"]
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
            gate = data["f1_gate"] if kind in ("source_aware", "rank", "arr_aux", "weather", "route_duration", "scheduled", "current_gate") else data["gate"]
            assert gate["passed"] is False, (kind, "gate changed")
            assert review.get("f1_gate_passed", review.get("gate_passed", False)) is False
            if kind == "route_duration":
                assert review["reported_gate_reproduced"] and review["maximum_metric_absolute_delta"] == 0
                assert review["verified_panel_count"] == 9
                for month, part in data["months"].items():
                    assert part["arms"]["raw"] == part["arms"]["clean"], (kind, month, "raw/clean scores differ")
                    assert part["comparisons"]["raw"] == part["comparisons"]["clean"], (kind, month, "raw/clean comparisons differ")
                    hashes = review["months"][month]["panels_sha256"]
                    assert hashes["raw"] == hashes["clean"], (kind, month, "raw/clean panels differ")
            if kind == "scheduled":
                assert data["status"] == review["decision"] == "stopped_after_F1"
                assert review["complete_panel_rmse_count"] == 9 and review["paired_comparison_count"] == 6
                assert review["maximum_absolute_reported_numeric_delta"] == 0
                assert review["admission_sha256"] == data["prescore_admission_sha256"]
                assert review["target_source_sha256"] == data["pins"]["target_source_sha256"]
                assert data["ranking_prediction"] is False and data["upload"] is False
                assert review["ranking_prediction"] is False and review["upload"] is False
                for matched_arm in ("landed", "clean"):
                    checks = gate["checks"][matched_arm]
                    assert all(checks[key] is False for key in (
                        "july_complete_gain_at_least_5_sec", "july_paired_day_ci95_lower_positive",
                        "july_each_day_removal_positive", "july_top10_beneficial_removal_positive"))
                    assert checks["december_regression_at_most_1_sec"] is True
                for monthly in data["months"].values():
                    assert set(monthly["arms"]) == {"scheduled", "landed", "clean"}
                    assert set(monthly["comparisons"]) == {"landed", "clean"}
                    for arm in monthly["arms"].values():
                        assert arm["rows"] == monthly["rows"]
                        assert math.isclose(arm["rmse"], math.sqrt(arm["sse"] / arm["rows"]),
                                            abs_tol=1e-7, rel_tol=0)
                    for matched_arm, comparison in monthly["comparisons"].items():
                        assert comparison["rows"] == monthly["rows"]
                        assert abs(comparison["candidate_rmse"] - monthly["arms"]["scheduled"]["rmse"]) < 1e-7
                        assert abs(comparison["comparator_rmse"] - monthly["arms"][matched_arm]["rmse"]) < 1e-7
                        assert abs((comparison["comparator_rmse"] - comparison["candidate_rmse"]) - comparison["gain_sec"]) < 1e-7
            if kind == "current_gate":
                assert data["status"] == review["decision"] == "stopped_after_F1"
                assert review["status"] == "independent_current_gate_f1_postscore_arithmetic_replayed_v1"
                assert review["complete_panel_rmse_count"] == 9 and review["paired_comparison_count"] == 6
                assert review["maximum_absolute_reported_numeric_delta"] == 0
                assert review["admission_sha256"] == data["prescore_admission_sha256"]
                assert review["target_source_sha256"] == data["pins"]["target_source_sha256"]
                assert data["ranking_prediction"] is False and data["upload"] is False
                assert review["ranking_prediction"] is False and review["upload"] is False
                for matched_arm in ("intercept", "clean"):
                    checks = gate["checks"][matched_arm]
                    assert checks["july_complete_gain_at_least_5_sec"] is False
                    assert all(checks[key] is True for key in (
                        "july_paired_day_ci95_lower_positive", "july_each_day_removal_positive",
                        "july_top10_beneficial_removal_positive", "december_regression_at_most_1_sec"))
                for monthly in data["months"].values():
                    assert set(monthly["arms"]) == {"contextual", "intercept", "clean"}
                    assert set(monthly["comparisons"]) == {"intercept", "clean"}
                    for arm in monthly["arms"].values():
                        assert arm["rows"] == monthly["rows"]
                        assert math.isclose(arm["rmse"], math.sqrt(arm["sse"] / arm["rows"]),
                                            abs_tol=1e-7, rel_tol=0)
                    for matched_arm, comparison in monthly["comparisons"].items():
                        assert comparison["rows"] == monthly["rows"]
                        assert abs(comparison["candidate_rmse"] - monthly["arms"]["contextual"]["rmse"]) < 1e-7
                        assert abs(comparison["comparator_rmse"] - monthly["arms"][matched_arm]["rmse"]) < 1e-7
                        assert abs((comparison["comparator_rmse"] - comparison["candidate_rmse"]) - comparison["gain_sec"]) < 1e-7
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
    src, src_hash, audit, audit_hash = ARR_DIAGNOSTIC
    data = json.loads(verified_bytes(src, src_hash))
    review = json.loads(verified_bytes(audit, audit_hash))
    assert data["status"] == "complete_october_november_diagnostic_scored"
    assert data["all_gates_pass"] is False
    assert review["score_report_sha256"] == src_hash
    assert review["status"] == "independent_complete_diagnostic_score_replay_pass"
    assert review["all_gates_pass"] is False
    assert review["ranking_target_read"] is False and review["uploaded"] is False
    for key, (month, count) in ARR_ROWS.items():
        part = data["months"][key]["comparisons"]["clean"]
        audited = review["months"][key]["comparisons"]["clean"]
        assert part == audited and part["rows"] == count
        assert part["passes_one_second_gate"] is False
        assert abs(part["control_rmse"] - part["candidate_rmse"] - part["gain_sec"]) < 1e-7
        result.append(dict(panel="retrospective_ARR_2025", direction="ARR innovation residual",
                           month=month, comparator="unchanged clean",
                           control_rmse_s=f"{part['control_rmse']:.9f}",
                           candidate_rmse_s=f"{part['candidate_rmse']:.9f}",
                           gain_s=f"{part['gain_sec']:.9f}", rows=count,
                           gate="DIAGNOSTIC_STOP", source=src, source_sha256=src_hash,
                           audit=audit, audit_sha256=audit_hash))
    return result


def draw(rows: list[dict]) -> None:
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(12.6, 9.5), constrained_layout=False)
    outer = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[6.5, 2.3],
                             width_ratios=[1, 1], left=.30, right=.965, top=.84,
                             bottom=.088, hspace=.65, wspace=.18)
    fig.text(.045, .963, "Campaign score evidence", fontsize=19, weight="bold", color="#172d39")
    fig.text(.045, .927, "Every matched F1 direction stopped before F3 or release. Positive paired gain is not qualification.",
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
    fig.text(.045, .024, f"Local bars: same-origin control minus candidate. All {len(SOURCES)} F1 gates failed."
             "  |  Source SHA-256s and comparator names: aggregate_scores.csv", fontsize=8.3, color="#53626b")
    fig.savefig(OUT / "campaign_scores.png", dpi=155)
    plt.close(fig)


def draw_arr() -> None:
    src, src_hash, _, _ = ARR_DIAGNOSTIC
    data = json.loads(verified_bytes(src, src_hash))
    fig, ax = plt.subplots(figsize=(8.0, 3.3))
    fig.subplots_adjust(left=.21, right=.94, top=.72, bottom=.25)
    ax.axvline(0, color="#77838b", linewidth=1)
    for y, (key, _) in enumerate(ARR_ROWS.items()):
        part = data["months"][key]["comparisons"]["clean"]
        gain = part["gain_sec"]
        lower, upper = part["paired_utc_day_bootstrap_ci95"]
        ax.barh(y, gain, color="#bd663c", height=.45, zorder=2)
        ax.errorbar(gain, y, xerr=[[gain - lower], [upper - gain]], fmt="none",
                    capsize=4, color="#172d39", linewidth=1.2, zorder=3)
        ax.text(.04, y, f"{gain:+.3f} s", ha="left", va="center",
                fontsize=10, color="#172d39")
    ax.set(yticks=[0, 1], yticklabels=["October 2025", "November 2025"],
           xlim=(-1.5, .25), ylim=(-.6, 1.6),
           xlabel="Unchanged clean minus ARR-corrected RMSE (seconds)")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#dde3e7", linewidth=.7, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.text(.035, .94, "ARR innovation residual: stopped diagnostic", fontsize=14,
             weight="bold", color="#172d39")
    fig.text(.035, .85, "Complete 2025 cohorts; whiskers are paired-day 95% intervals.",
             fontsize=9.6, color="#53626b")
    fig.text(.035, .055, "Separate fit history from the F1 and official panels. Negative gain means worse RMSE.",
             fontsize=8.7, color="#53626b")
    fig.savefig(OUT / "arr_diagnostic.png", dpi=155, facecolor="white")
    plt.close(fig)


def main() -> None:
    rows = aggregate_rows()
    with (OUT / "aggregate_scores.csv").open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    draw(rows)
    draw_arr()
    print(f"Verified {len(SOURCES)} matched F1 directions, {len(OFFICIAL)} official receipts, "
          f"and {len(ARR_ROWS)} ARR diagnostic months; wrote {len(rows)} aggregate rows.")


if __name__ == "__main__":
    main()
