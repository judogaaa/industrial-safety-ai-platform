"""Evaluate the detector over a folder of images.

Usage (repo root, .venv active):
    python backend/scripts/evaluate_folder.py data/samples [--labels labels.json] [--runs 1]

Without --labels reports honest detector statistics only (no ground truth
is invented): frames, latency/FPS, detections per class, mean/min confidence,
borderline ("weak") detections, unassigned PPE, worker count.

Labels JSON: {"images": {"name.jpg": [{"class": "person", "bbox": [x1,y1,x2,y2]}]}}
Labels CSV: header image,class,x1,y1,x2,y2. Coordinates normalized 0..1.
With labels additionally reports per-class precision/recall at IoU 0.5.

With --events EVENTS.json runs the full incident pipeline (tracker +
TemporalEngine consensus) over frames in filename order and compares
detected incidents against ground-truth events:
  {"events": [{"type": "no_helmet", "start_frame": 10, "end_frame": 50}]}
Reports matched/missed/false-positive incidents, detection-to-alert latency
in frames, and worker-ID stability proxy (distinct IDs per clip).

Worker-aware GT (recommended): each event names a stable worker id and
gives sparse worker boxes so an alarm on the WRONG person never counts:
  {"events": [{"type": "no_helmet", "worker": "w1",
               "start_frame": 10, "end_frame": 50,
               "boxes": {"10": [x1,y1,x2,y2], "30": [...]}}]}
Coordinates normalized 0..1. Events without boxes fall back to time-only
matching and are flagged worker_unverified in the report.
"""

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0


def load_labels(path: Path) -> dict:
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return data.get("images", data)
    rows: dict[str, list] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.setdefault(row["image"], []).append(
                {
                    "class": row["class"],
                    "bbox": [float(row[k]) for k in ("x1", "y1", "x2", "y2")],
                }
            )
    return rows


def main():
    from app.core.config import settings
    from app.inference.local_onnx import LocalOnnxProvider
    from app.services.engine import Tracker, assign_detections
    from app.services.schemas import CameraConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--events", default=None,
                        help="GT events JSON for incident-level scoring")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Frame timestamps step for the incident pipeline")
    parser.add_argument("--report-json", default=None,
                        help="Write machine-readable incident evaluation results")
    parser.add_argument("--fully-annotated", action="store_true",
                        help="Assert all workers and violation episodes are labelled")
    parser.add_argument("--min-worker-recall", type=float, default=None,
                        help="Fail evaluation below this worker-verified recall (0-1)")
    parser.add_argument("--max-false-alarms-per-hour", type=float, default=None,
                        help="Fail evaluation above this observed false-alarm rate")
    parser.add_argument("--window", type=int, default=None,
                        help="Temporal consensus window (default: camera config)")
    parser.add_argument("--required", type=int, default=None,
                        help="Required positive votes in window (default: camera config)")
    parser.add_argument("--confirm", type=float, default=2.0,
                        help="Seconds of evidence before ACTIVE")
    parser.add_argument("--clear", type=float, default=2.0,
                        help="Seconds of counter-evidence before RESOLVED")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be greater than zero")
    if args.report_json and not args.events:
        parser.error("--report-json requires --events")
    if (args.min_worker_recall is not None
            or args.max_false_alarms_per_hour is not None):
        if not args.events or not args.fully_annotated:
            parser.error("quality gates require --events and --fully-annotated")
    if args.min_worker_recall is not None and not 0 <= args.min_worker_recall <= 1:
        parser.error("--min-worker-recall must be between 0 and 1")
    if args.max_false_alarms_per_hour is not None and args.max_false_alarms_per_hour < 0:
        parser.error("--max-false-alarms-per-hour must be non-negative")

    images = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in EXTS)
    if not images:
        raise SystemExit(f"No images in {args.folder}")
    labels = load_labels(Path(args.labels)) if args.labels else {}

    import cv2

    provider = LocalOnnxProvider(
        str(settings.resolve(settings.model_path)),
        settings.confidence_threshold,
        settings.inference_threads,
        person_model_path=(
            str(settings.resolve(settings.person_model_path))
            if settings.use_separate_person_model
            else None
        ),
        providers=settings.onnx_provider_list,
        thresholds=settings.class_thresholds,
    )
    config = CameraConfig(helmet=True, vest=True, mask=True, smoking=True)
    tracker = Tracker()
    lat: list[float] = []
    per_class: Counter = Counter()
    conf_sum: dict[str, float] = {}
    weak = 0
    unassigned = 0
    workers: set[int] = set()
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    for step, path in enumerate(images):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"Skip undecodable: {path.name}")
            continue
        started = time.perf_counter()
        detections = provider.predict_sync(frame)
        lat.append((time.perf_counter() - started) * 1000)
        tracks = tracker.update(detections, frame_timestamp(step, args.fps))
        _, owners = assign_detections(tracks, detections, config)
        workers.update(t.id for t in tracks)
        for index, det in enumerate(detections):
            per_class[det.class_name] += 1
            conf_sum[det.class_name] = conf_sum.get(det.class_name, 0) + det.confidence
            limit = provider.threshold_for(det.class_name)
            if det.confidence < limit + 0.10:
                weak += 1
            if det.class_name != "person" and owners.get(index) is None:
                unassigned += 1
        expected = labels.get(path.name, [])
        matched = set()
        for det in detections:
            if det.weak:
                # Weak diagnostics never count as confirmed detections (T3).
                continue
            box = [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]
            best, best_iou = None, 0.5
            for i, item in enumerate(expected):
                if i in matched or item["class"] != det.class_name:
                    continue
                value = iou(box, item["bbox"])
                if value >= best_iou:
                    best, best_iou = i, value
            if best is None:
                fp[det.class_name] += 1
            else:
                matched.add(best)
                tp[det.class_name] += 1
        for i, item in enumerate(expected):
            if i not in matched:
                fn[item["class"]] += 1

    if not lat:
        raise SystemExit("No decodable frames; cannot report latency or accuracy")

    print(f"Frames: {len(images)}, workers seen: {len(workers)}")
    print(f"Latency: mean {statistics.mean(lat):.0f} ms, "
          f"p95 {sorted(lat)[int(len(lat) * 0.95)]:.0f} ms, "
          f"{1000 / statistics.mean(lat):.1f} FPS")
    print("Detections/class:", dict(sorted(per_class.items())))
    print("Mean confidence: " + ", ".join(
        f"{k}={conf_sum[k] / per_class[k]:.2f}" for k in sorted(per_class)
    ))
    print(f"Weak (within 0.10 above threshold): {weak}, unassigned PPE: {unassigned}")
    if labels:
        print("Per-class P/R @ IoU 0.5 (labels provided):")
        for cls in sorted(set(tp) | set(fp) | set(fn)):
            precision = tp[cls] / max(1, tp[cls] + fp[cls])
            recall = tp[cls] / max(1, tp[cls] + fn[cls])
            print(f"  {cls}: P={precision:.2f} R={recall:.2f} "
                  f"(tp={tp[cls]} fp={fp[cls]} fn={fn[cls]})")
    if args.events:
        report = score_events(images, provider, args)
        if args.report_json:
            output_path = Path(args.report_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Evaluation report: {output_path}")
        metrics = report["metrics"]
        if args.min_worker_recall is not None:
            if report["ground_truth"]["with_worker_boxes"] != report["ground_truth"]["total"]:
                raise SystemExit("Cannot gate worker recall: GT episodes lack worker boxes")
            recall = metrics["worker_verified_recall"]
            if recall is None:
                raise SystemExit("Cannot gate worker recall: no annotated violations")
            if recall < args.min_worker_recall:
                raise SystemExit(
                    f"FAIL worker recall {recall:.3f} < {args.min_worker_recall:.3f}"
                )
        if args.max_false_alarms_per_hour is not None:
            if report["ground_truth"]["with_worker_boxes"] != report["ground_truth"]["total"]:
                raise SystemExit("Cannot gate false alarms: GT episodes lack worker boxes")
            if metrics["worker_unverified"]:
                raise SystemExit("Cannot gate false alarms: some detections lack worker verification")
            if metrics["observed_false_alarms_per_hour"] > args.max_false_alarms_per_hour:
                raise SystemExit(
                    "FAIL false alarms/hour "
                    f"{metrics['observed_false_alarms_per_hour']:.3f} > "
                    f"{args.max_false_alarms_per_hour:.3f}"
                )


def _norm_boxes(boxes) -> dict[int, list[float]]:
    out = {}
    for k, v in (boxes or {}).items():
        try:
            out[int(k)] = [float(x) for x in v]
        except (ValueError, TypeError):
            continue
    return out


def _spatial_agreement(det_boxes: dict[int, list[float]],
                       gt_boxes: dict[int, list[float]],
                       start: int, end: int) -> tuple[bool, bool]:
    """Worker linkage between a detected incident and a GT event.

    Returns (agree, verifiable). Compares track boxes vs GT worker boxes on
    common annotated frames within the overlap: >=50% with IoU>=0.3 agrees.
    No common frames -> not verifiable (caller falls back to time-only and
    flags the match worker_unverified instead of inventing an identity).
    """
    frames = [f for f in gt_boxes if start <= f <= end and f in det_boxes]
    if not frames:
        return False, False
    hits = sum(1 for f in frames if iou(det_boxes[f], gt_boxes[f]) >= 0.3)
    return hits / len(frames) >= 0.5, True


def frame_timestamp(step: int, fps: float) -> float:
    """Source time; detection/tracking and event scoring must use the same FPS."""
    if fps <= 0:
        raise ValueError("FPS must be positive")
    return step / fps


def match_incidents(detected: list[dict], gt_events: list[dict]) -> dict:
    """One-to-one event matching that prioritizes verified worker identity.

    Maximize spatially verified matches first, then match remaining time-only
    candidates. A temporal overlap must never displace an available verified
    worker match. Neither pass can reuse a GT event or detected incident.
    """
    verified: dict[int, list[int]] = {i: [] for i in range(len(detected))}
    time_only: dict[int, list[int]] = {i: [] for i in range(len(detected))}
    mismatches: set[int] = set()

    for di, det in enumerate(detected):
        det_boxes = _norm_boxes(det.get("boxes"))
        for gi, event in enumerate(gt_events):
            if event.get("type") != det.get("type"):
                continue
            start = max(det["start_frame"], event["start_frame"])
            end = min(det["end_frame"], event["end_frame"])
            if start > end:
                continue
            gt_boxes = _norm_boxes(event.get("boxes"))
            if not gt_boxes:
                time_only[di].append(gi)
                continue
            agree, verifiable = _spatial_agreement(det_boxes, gt_boxes, start, end)
            if agree:
                verified[di].append(gi)
            elif verifiable:
                mismatches.add(di)
            else:
                time_only[di].append(gi)

    def augment(
        di: int, edges: dict[int, list[int]], owners: dict[int, int], seen: set[int]
    ) -> bool:
        """Find a one-to-one assignment, including reassignment of earlier hits."""
        for gi in edges[di]:
            if gi in seen:
                continue
            seen.add(gi)
            incumbent = owners.get(gi)
            if incumbent is None or augment(incumbent, edges, owners, seen):
                owners[gi] = di
                return True
        return False

    verified_owner: dict[int, int] = {}
    for di in range(len(detected)):
        augment(di, verified, verified_owner, set())

    verified_dets = set(verified_owner.values())
    available_gt = set(range(len(gt_events))) - set(verified_owner)
    remaining_edges = {
        di: [gi for gi in time_only[di] if gi in available_gt]
        for di in range(len(detected)) if di not in verified_dets
    }
    time_only_owner: dict[int, int] = {}
    for di in remaining_edges:
        augment(di, remaining_edges, time_only_owner, set())

    assignments = {di: (gi, False) for gi, di in verified_owner.items()}
    assignments.update({di: (gi, True) for gi, di in time_only_owner.items()})
    matched = [(di, assignments[di][0]) for di in sorted(assignments)]
    taken = {gi for _, gi in matched}
    return {
        "matched": matched,
        "missed": [gi for gi in range(len(gt_events)) if gi not in taken],
        "false_alarms": [di for di in range(len(detected)) if di not in assignments],
        "latencies": [
            detected[di]["start_frame"] - gt_events[gi]["start_frame"]
            for di, gi in matched
        ],
        "worker_mismatch": [di for di in sorted(mismatches) if di not in assignments],
        "unverified": [
            di for di, (_, unverified) in sorted(assignments.items()) if unverified
        ],
    }


def incident_quality_report(
    detected: list[dict],
    gt_events: list[dict],
    result: dict,
    fps: float,
    frame_count: int,
    fully_annotated: bool = False,
) -> dict:
    """Portable, conservative quality metrics for an independently labelled clip.

    Strict recall counts only matches with worker spatial evidence; time-only
    episodes are not credited as verified true positives. Metrics are not
    claims of physical safety or of generalization beyond this clip.
    """
    if fps <= 0 or frame_count <= 0:
        raise ValueError("Evaluation requires positive FPS and a nonempty clip")
    unverified = set(result["unverified"])
    verified_pairs = [(di, gi) for di, gi in result["matched"] if di not in unverified]
    classes = sorted(set(e["type"] for e in gt_events) | set(d["type"] for d in detected))
    by_type = {}
    for kind in classes:
        gt_ids = {i for i, e in enumerate(gt_events) if e["type"] == kind}
        det_ids = {i for i, e in enumerate(detected) if e["type"] == kind}
        by_type[kind] = {
            "ground_truth": len(gt_ids),
            "detected": len(det_ids),
            "verified_matches": sum(gi in gt_ids for _, gi in verified_pairs),
            "unverified_matches": sum(di in det_ids for di in result["unverified"]),
            "missed": sum(gi in gt_ids for gi in result["missed"]),
            "false_alarms": sum(di in det_ids for di in result["false_alarms"]),
        }

    duration = frame_count / fps
    return {
        "schema_version": 1,
        "frames": frame_count,
        "fps": fps,
        "duration_seconds": round(duration, 3),
        "fully_annotated_claim": fully_annotated,
        "ground_truth": {
            "total": len(gt_events),
            "with_worker_boxes": sum(
                bool(_norm_boxes(event.get("boxes"))) for event in gt_events
            ),
        },
        "metrics": {
            "detected": len(detected),
            "worker_verified_tp": len(verified_pairs),
            "time_overlap_matches": len(result["matched"]),
            "worker_verified_recall": (
                round(len(verified_pairs) / len(gt_events), 6)
                if gt_events else None
            ),
            "missed": len(result["missed"]),
            "false_alarms": len(result["false_alarms"]),
            "observed_false_alarms_per_hour": round(
                len(result["false_alarms"]) * 3600 / duration, 3
            ),
            "worker_mismatch": len(result["worker_mismatch"]),
            "worker_unverified": len(result["unverified"]),
        },
        "by_type": by_type,
        "limitations": [
            "Worker-verified recall is undefined when no violations are labelled.",
            "False alarms/hour is meaningful only if the entire clip is annotated.",
            "Model accuracy on this clip does not establish industrial safety.",
        ],
    }


def score_events(images, provider, args):
    """Incident-level scoring: GT events vs detected ACTIVE incidents."""
    import cv2

    from app.services.engine import (
        TemporalEngine,
        Tracker,
        assign_detections,
        cleared_assigned,
        evaluate_assigned,
    )
    from app.services.schemas import CameraConfig

    gt = json.loads(Path(args.events).read_text()).get("events", [])
    config = CameraConfig(helmet=True, vest=True, mask=True, smoking=True)
    window = args.window or config.temporal_window
    required = args.required or config.temporal_required
    tracker = Tracker()
    engine = TemporalEngine()
    detected: list[dict] = []
    open_incidents: dict = {}
    id_frames: dict[int, int] = {}
    track_boxes: dict[int, dict[int, list[float]]] = {}
    for step, path in enumerate(images):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        now = frame_timestamp(step, args.fps)
        detections = provider.predict_sync(frame)
        tracks = tracker.update(detections, now)
        for track in tracks:
            id_frames[track.id] = id_frames.get(track.id, 0) + 1
            track_boxes.setdefault(track.id, {})[step] = [
                track.bbox.x1, track.bbox.y1, track.bbox.x2, track.bbox.y2]
        assignments, _ = assign_detections(tracks, detections, config)
        observed = evaluate_assigned(tracks, assignments, config)
        cleared = cleared_assigned(tracks, assignments, config)
        for action, key, _state in engine.update(
            observed, now, args.confirm, args.clear, window, required, cleared,
        ):
            if action == "active":
                open_incidents[key] = step
            elif action in {"resolved", "lost", "interrupted"} and key in open_incidents:
                start = open_incidents.pop(key)
                detected.append({
                    "type": key[1], "start_frame": start,
                    "end_frame": step, "outcome": action, "worker": key[0],
                    "boxes": {f: b for f, b in track_boxes.get(key[0], {}).items()
                              if start <= f <= step},
                })
    for key, start in open_incidents.items():
        detected.append({"type": key[1], "start_frame": start,
                         "end_frame": len(images) - 1, "outcome": "active",
                         "worker": key[0],
                         "boxes": {f: b for f, b in track_boxes.get(key[0], {}).items()
                                   if f >= start}})
    res = match_incidents(detected, gt)
    latencies = res["latencies"]
    duration_hours = len(images) / args.fps / 3600
    unverified_det = set(res["unverified"])
    verified = [m for m in res["matched"] if m[0] not in unverified_det]
    print(f"\nIncident-level scoring ({len(gt)} GT events, {len(detected)} detected):")
    print(f"  params: window={window} required={required} "
          f"confirm={args.confirm}s clear={args.clear}s")
    print(f"  matched={len(res['matched'])} missed={len(res['missed'])} "
          f"false_alarms={len(res['false_alarms'])}")
    print(f"  observed false alarms/hour={len(res['false_alarms']) / duration_hours:.2f} "
          "(meaningful only when the entire clip is fully annotated)")
    if res["worker_mismatch"]:
        print(f"  worker_mismatch (alarm on wrong person)={len(res['worker_mismatch'])}: "
              f"{res['worker_mismatch']}")
    if res["unverified"]:
        print(f"  worker_unverified (no GT boxes, time-only)={len(res['unverified'])}: "
              f"{res['unverified']}")
    print(f"  strict worker-verified matches={len(verified)} "
          f"(time-only matches excluded from the strict metric)")
    if gt:
        print(f"  recall: worker-verified {len(verified)}/{len(gt)} = "
              f"{len(verified) / len(gt):.3f}; "
              f"time-only {len(res['matched'])}/{len(gt)} = "
              f"{len(res['matched']) / len(gt):.3f}")
    else:
        print("  recall: N/A (no labelled violation episodes)")
    if latencies:
        print(f"  detection-to-alert: mean {statistics.mean(latencies):.1f} frames, "
              f"max {max(latencies)} frames")
    print(f"  worker-ID stability proxy: {len(id_frames)} distinct IDs, "
          f"frames/ID mean {statistics.mean(id_frames.values()):.1f}")
    return incident_quality_report(
        detected, gt, res, args.fps, len(images), args.fully_annotated
    )


if __name__ == "__main__":
    main()
