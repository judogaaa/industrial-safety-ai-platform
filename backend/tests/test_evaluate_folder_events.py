"""Track-aware incident matching for evaluate_folder --events.

Core regression: an alarm on the WRONG person must never score as a match,
even with identical type and overlapping time.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

_spec = importlib.util.spec_from_file_location(
    "evaluate_folder", ROOT / "backend" / "scripts" / "evaluate_folder.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def det(typ, start, end, boxes=None, worker=1):
    return {"type": typ, "start_frame": start, "end_frame": end,
            "worker": worker, "boxes": boxes or {}}


def gt(typ, start, end, worker="w1", boxes=None):
    e = {"type": typ, "start_frame": start, "end_frame": end, "worker": worker}
    if boxes is not None:
        e["boxes"] = boxes
    return e


BOX_A = [0.1, 0.1, 0.4, 0.6]
BOX_B = [0.6, 0.1, 0.9, 0.6]  # far from A: IoU 0


def test_match_same_worker():
    d = [det("no_helmet", 10, 50, {10: BOX_A, 30: BOX_A, 50: BOX_A})]
    g = [gt("no_helmet", 12, 48, "w1", {"12": BOX_A, "30": BOX_A})]
    r = ev.match_incidents(d, g)
    assert r["matched"] == [(0, 0)]
    assert r["missed"] == [] and r["false_alarms"] == []
    assert r["latencies"] == [-2]
    assert r["worker_mismatch"] == [] and r["unverified"] == []


def test_wrong_worker_never_matches():
    d = [det("no_helmet", 10, 50, {10: BOX_B, 30: BOX_B})]
    g = [gt("no_helmet", 12, 48, "w1", {"12": BOX_A, "30": BOX_A})]
    r = ev.match_incidents(d, g)
    assert r["matched"] == []
    assert r["missed"] == [0]
    assert r["false_alarms"] == [0]
    assert r["worker_mismatch"] == [0]


def test_legacy_gt_without_boxes_time_only_unverified():
    d = [det("no_vest", 5, 20, {5: BOX_A})]
    g = [gt("no_vest", 6, 18, "w1")]
    r = ev.match_incidents(d, g)
    assert r["matched"] == [(0, 0)]
    assert r["unverified"] == [0]
    assert r["worker_mismatch"] == []


def test_no_common_box_frames_falls_back_unverified():
    d = [det("no_helmet", 10, 50, {40: BOX_B})]
    g = [gt("no_helmet", 12, 48, "w1", {"12": BOX_A})]
    r = ev.match_incidents(d, g)
    assert r["matched"] == [(0, 0)]
    assert r["unverified"] == [0]


def test_one_to_one_and_type_mismatch():
    d = [det("no_helmet", 10, 20, {10: BOX_A}),
         det("no_helmet", 12, 22, {12: BOX_A}),
         det("no_vest", 10, 20, {10: BOX_A})]
    g = [gt("no_helmet", 11, 19, "w1", {"11": BOX_A})]
    r = ev.match_incidents(d, g)
    assert r["matched"] == [(0, 0)]
    assert r["false_alarms"] == [1, 2]
    assert r["missed"] == []


def test_strict_metric_excludes_time_only():
    d = [det("no_vest", 5, 20, {5: BOX_A})]
    g = [gt("no_vest", 6, 18, "w1")]
    r = ev.match_incidents(d, g)
    assert r["matched"] == [(0, 0)]
    unverified = set(r["unverified"])
    verified = [m for m in r["matched"] if m[0] not in unverified]
    assert verified == []  # time-only must NOT count as worker-verified
    d2 = [det("no_vest", 5, 20, {6: BOX_A, 10: BOX_A})]
    g2 = [gt("no_vest", 6, 18, "w1", {"6": BOX_A, "10": BOX_A})]
    r2 = ev.match_incidents(d2, g2)
    verified2 = [m for m in r2["matched"] if m[0] not in set(r2["unverified"])]
    assert verified2 == [(0, 0)]


def test_spatial_agreement_majority():
    agree, ver = ev._spatial_agreement(
        {1: BOX_A, 2: BOX_A, 3: BOX_B}, {1: BOX_A, 2: BOX_A, 3: BOX_A}, 1, 3)
    assert (agree, ver) == (True, True)
    agree, ver = ev._spatial_agreement(
        {1: BOX_B, 2: BOX_B, 3: BOX_A}, {1: BOX_A, 2: BOX_A, 3: BOX_A}, 1, 3)
    assert (agree, ver) == (False, True)
    agree, ver = ev._spatial_agreement({99: BOX_A}, {1: BOX_A}, 1, 50)
    assert (agree, ver) == (False, False)



def test_spatially_verified_match_beats_earlier_unverified_alarm():
    # A time-only overlap must not consume the only worker-verified GT.
    d = [
        det("no_helmet", 10, 30, {10: BOX_B}),
        det("no_helmet", 10, 30, {20: BOX_A}),
    ]
    g = [gt("no_helmet", 10, 30, "w1", {"20": BOX_A})]
    r = ev.match_incidents(d, g)
    assert r["matched"] == [(1, 0)]
    assert r["unverified"] == []
    assert r["false_alarms"] == [0]


def test_bipartite_matching_recovers_both_verified_events():
    # First alarm can fit either GT; second alarm can fit only the first.
    # Naive first-hit greedy matching misses the second GT entirely.
    d = [
        det("no_vest", 0, 50, {10: BOX_A, 20: BOX_A}),
        det("no_vest", 10, 19, {10: BOX_A}),
    ]
    g = [
        gt("no_vest", 0, 20, "w1", {"10": BOX_A}),
        gt("no_vest", 20, 50, "w2", {"20": BOX_A}),
    ]
    r = ev.match_incidents(d, g)
    assert r["matched"] == [(0, 1), (1, 0)]
    assert r["missed"] == r["false_alarms"] == r["unverified"] == []


def test_time_only_matching_cannot_evict_verified_worker():
    d = [
        det("no_helmet", 10, 30, {20: BOX_A}),
        det("no_helmet", 10, 30),
    ]
    g = [
        gt("no_helmet", 10, 30, "unknown"),
        gt("no_helmet", 10, 30, "w1", {"20": BOX_A}),
    ]
    r = ev.match_incidents(d, g)
    assert r["matched"] == [(0, 1), (1, 0)]
    assert r["unverified"] == [1]
    assert r["false_alarms"] == []


def test_spatial_mismatch_is_never_a_verified_match():
    d = [det("no_helmet", 10, 30, {20: BOX_B})]
    g = [gt("no_helmet", 10, 30, "w1", {"20": BOX_A})]
    r = ev.match_incidents(d, g)
    assert r["matched"] == []
    assert r["missed"] == [0]
    assert r["false_alarms"] == [0]
    assert r["worker_mismatch"] == [0]


def test_frame_timestamp_uses_configured_fps_instead_of_hardcoded_five():
    assert ev.frame_timestamp(10, 10.0) == 1.0
    assert ev.frame_timestamp(10, 5.0) == 2.0
    assert abs(ev.frame_timestamp(10, 7.0) - 10 / 7.0) < 1e-12


def test_zero_fps_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="FPS must be positive"):
        ev.frame_timestamp(10, 0.0)


def test_unlabelled_detected_event_is_false_alarm_not_a_match():
    r = ev.match_incidents([det("no_vest", 1, 20)], [])
    assert r["matched"] == []
    assert r["false_alarms"] == [0]
    assert r["missed"] == []



def test_machine_report_clean_clip_has_undefined_recall_and_real_rate():
    detected = [det("no_helmet", 2, 3)]
    result = ev.match_incidents(detected, [])
    report = ev.incident_quality_report(detected, [], result, fps=5, frame_count=10,
                                        fully_annotated=True)
    assert report["metrics"]["worker_verified_recall"] is None
    assert report["metrics"]["false_alarms"] == 1
    assert report["metrics"]["observed_false_alarms_per_hour"] == 1800.0
    assert report["ground_truth"]["total"] == 0
    assert report["fully_annotated_claim"]


def test_machine_report_counts_verified_vs_time_only_by_class():
    detected = [
        det("no_helmet", 10, 30, {20: BOX_A}),
        det("no_vest", 10, 30),
    ]
    ground_truth = [
        gt("no_helmet", 10, 30, "w1", {"20": BOX_A}),
        gt("no_vest", 10, 30, "w2"),
    ]
    matched = ev.match_incidents(detected, ground_truth)
    report = ev.incident_quality_report(
        detected, ground_truth, matched, fps=10.0, frame_count=100
    )
    assert report["metrics"]["worker_verified_tp"] == 1
    assert report["metrics"]["time_overlap_matches"] == 2
    assert report["metrics"]["worker_verified_recall"] == 0.5
    assert report["metrics"]["worker_unverified"] == 1
    assert report["by_type"]["no_helmet"]["verified_matches"] == 1
    assert report["by_type"]["no_vest"]["verified_matches"] == 0
    assert report["by_type"]["no_vest"]["unverified_matches"] == 1
    assert report["ground_truth"]["with_worker_boxes"] == 1


def test_machine_report_rejects_impossible_timeline():
    import pytest

    result = ev.match_incidents([], [])
    with pytest.raises(ValueError, match="positive FPS"):
        ev.incident_quality_report([], [], result, 0, 10)
    with pytest.raises(ValueError, match="nonempty clip"):
        ev.incident_quality_report([], [], result, 5, 0)


def test_machine_report_json_serializes_clean_gt_without_inventing_recall():
    import json

    result = ev.match_incidents([], [])
    report = ev.incident_quality_report([], [], result, 5.0, 10)
    encoded = json.dumps(report, ensure_ascii=False)
    decoded = json.loads(encoded)
    assert decoded["metrics"]["worker_verified_recall"] is None
    assert decoded["metrics"]["observed_false_alarms_per_hour"] == 0
