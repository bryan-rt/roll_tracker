"""Audio landmark detector — survey tool for buzzer/bell detection in gym footage.

Extracts audio events from MP4 clips and emits audio_events.jsonl per clip.
Three event classes: sustained_tone (buzzer/bell), sharp_transient (starting bell),
noise_burst (everything else above energy threshold).

Usage:
    python tools/detect_buzzer.py --input <mp4_or_dir> [--survey] [--output-dir <dir>]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def find_mp4s(input_path: Path) -> list[Path]:
    """Return list of MP4 paths from a single file or directory (excluding /diag/)."""
    if input_path.is_file():
        if input_path.suffix.lower() == ".mp4":
            return [input_path]
        return []
    if input_path.is_dir():
        return sorted(
            p for p in input_path.rglob("*.mp4")
            if "/diag/" not in str(p)
        )
    return []


def get_fps(mp4_path: Path) -> float:
    """Read FPS from MP4 via cv2. Falls back to 30.0."""
    cap = cv2.VideoCapture(str(mp4_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 30.0
    finally:
        cap.release()


def detect_events(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    min_duration_ms: float,
    min_peak_db: float,
) -> list[dict[str, Any]]:
    """Detect audio events from waveform. Returns list of event dicts sorted by timestamp."""
    import librosa
    from scipy.signal import find_peaks

    n_frames = 1 + len(y) // hop_length

    # Compute features
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # Energy threshold
    rms_mean = rms.mean()
    rms_std = rms.std()
    energy_thresh = rms_mean + 1.5 * rms_std
    flatness_thresh = 0.3

    def hop_to_ms(hop_idx: int) -> float:
        return hop_idx * hop_length / sr * 1000

    def dominant_freq_at(start_hop: int, end_hop: int) -> float:
        seg = S[:, start_hop:end_hop]
        if seg.size == 0:
            return 0.0
        mean_mag = seg.mean(axis=1)
        bin_idx = int(np.argmax(mean_mag))
        return float(freqs[bin_idx]) if bin_idx < len(freqs) else 0.0

    def peak_db_in(start_hop: int, end_hop: int) -> float:
        seg = rms[start_hop:end_hop]
        if seg.size == 0:
            return -100.0
        return float(librosa.amplitude_to_db(np.array([seg.max()]))[0])

    # Classify frames
    high_energy = rms > energy_thresh
    tonal = flatness < flatness_thresh
    min_frames = len(rms)

    # Merge contiguous frames into events for sustained_tone and noise_burst
    events: list[dict[str, Any]] = []

    def merge_and_emit(mask: np.ndarray, event_class: str) -> None:
        in_event = False
        start = 0
        for i in range(len(mask)):
            if mask[i] and not in_event:
                start = i
                in_event = True
            elif not mask[i] and in_event:
                _emit_merged(start, i, event_class)
                in_event = False
        if in_event:
            _emit_merged(start, len(mask), event_class)

    def _emit_merged(start_hop: int, end_hop: int, event_class: str) -> None:
        dur_ms = hop_to_ms(end_hop - start_hop)
        if dur_ms < min_duration_ms:
            return
        db = peak_db_in(start_hop, end_hop)
        if db < min_peak_db:
            return
        events.append({
            "timestamp_ms": round(hop_to_ms(start_hop), 1),
            "duration_ms": round(dur_ms, 1),
            "peak_db": round(db, 1),
            "dominant_freq_hz": round(dominant_freq_at(start_hop, end_hop), 1),
            "event_class": event_class,
        })

    # sustained_tone: high energy + tonal
    sustained_mask = high_energy & tonal
    merge_and_emit(sustained_mask, "sustained_tone")

    # noise_burst: high energy + not tonal
    noise_mask = high_energy & ~tonal
    merge_and_emit(noise_mask, "noise_burst")

    # sharp_transient: onset peaks
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_mean = onset_env.mean()
    onset_std = onset_env.std()
    onset_thresh = onset_mean + 2.5 * onset_std
    min_distance = int(sr / hop_length * 0.5)
    if min_distance < 1:
        min_distance = 1

    peaks, _ = find_peaks(onset_env, height=onset_thresh, distance=min_distance)
    hop_dur_ms = round(hop_length / sr * 1000, 1)

    for p in peaks:
        db = peak_db_in(p, p + 1)
        if db < min_peak_db:
            continue
        events.append({
            "timestamp_ms": round(hop_to_ms(p), 1),
            "duration_ms": hop_dur_ms,
            "peak_db": round(db, 1),
            "dominant_freq_hz": round(dominant_freq_at(p, p + 1), 1),
            "event_class": "sharp_transient",
        })

    events.sort(key=lambda e: e["timestamp_ms"])
    return events


def compute_periodicity(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute periodicity stats for sustained_tone events."""
    tones = [e for e in events if e["event_class"] == "sustained_tone"]
    result: dict[str, Any] = {
        "n_sustained_tone": len(tones),
        "sustained_tone_intervals_s": [],
        "sustained_tone_mean_interval_s": None,
        "sustained_tone_cv": None,
        "is_periodic": False,
    }
    if len(tones) < 2:
        return result

    timestamps_s = [t["timestamp_ms"] / 1000 for t in tones]
    intervals = [round(timestamps_s[i + 1] - timestamps_s[i], 1) for i in range(len(timestamps_s) - 1)]
    result["sustained_tone_intervals_s"] = intervals

    if len(intervals) >= 1:
        mean_iv = round(float(np.mean(intervals)), 1)
        std_iv = float(np.std(intervals))
        result["sustained_tone_mean_interval_s"] = mean_iv
        if mean_iv > 0 and len(tones) >= 3:
            cv = round(std_iv / mean_iv, 4)
            result["sustained_tone_cv"] = cv
            result["is_periodic"] = cv < 0.05

    return result


def format_timestamp(ms: float) -> str:
    """Format milliseconds as HH:MM:SS.t."""
    total_s = ms / 1000
    h = int(total_s // 3600)
    m = int((total_s % 3600) // 60)
    s = total_s % 60
    return f"{h:02d}:{m:02d}:{s:04.1f}"


def print_survey(clip_id: str, events: list[dict[str, Any]], periodicity: dict[str, Any]) -> None:
    """Print human-readable timeline and periodicity check to stdout."""
    print(f"\n=== Audio event timeline: {clip_id}.mp4 ===")
    if not events:
        print("  (no events detected)")
    else:
        for e in events:
            ts = format_timestamp(e["timestamp_ms"])
            print(
                f"  {ts}  {e['event_class']:<18s} {e['duration_ms']:>6.0f}ms"
                f"  {e['dominant_freq_hz']:>7.1f}Hz  {e['peak_db']:>6.1f}dB"
            )

    # Counts
    counts: dict[str, int] = {}
    for e in events:
        counts[e["event_class"]] = counts.get(e["event_class"], 0) + 1
    count_parts = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    print(f"\nTotal events: {len(events)}  ({count_parts})")

    # Periodicity
    if periodicity["n_sustained_tone"] >= 3:
        intervals = periodicity["sustained_tone_intervals_s"]
        mean_iv = periodicity["sustained_tone_mean_interval_s"]
        cv = periodicity["sustained_tone_cv"]
        label = "PERIODIC" if periodicity["is_periodic"] else "APERIODIC"
        print(f"\nPeriodicity check:")
        print(f"  sustained_tone intervals (s): {intervals}")
        print(f"  Mean interval: {mean_iv}s  Std: {round(float(np.std(intervals)), 1)}s  → {label} (CV={cv} {'<' if periodicity['is_periodic'] else '>='} 0.05)")
    elif periodicity["n_sustained_tone"] > 0:
        print(f"\nPeriodicity check: too few sustained_tone events ({periodicity['n_sustained_tone']}) — need 3+")
    print()


def process_clip(
    mp4_path: Path,
    output_dir: Path | None,
    survey: bool,
    min_duration_ms: float,
    min_peak_db: float,
    sr: int,
    hop_length: int,
) -> bool:
    """Process a single MP4 clip. Returns True on success."""
    import librosa

    clip_id = mp4_path.stem

    # Load audio
    try:
        y, loaded_sr = librosa.load(str(mp4_path), sr=sr, mono=True)
    except Exception as exc:
        log.warning("Skipping %s — failed to load audio: %s", mp4_path.name, exc)
        return False

    if len(y) == 0:
        log.warning("Skipping %s — empty audio stream", mp4_path.name)
        return False

    log.info("Processing %s (%.1fs audio at %d Hz)", clip_id, len(y) / loaded_sr, loaded_sr)

    fps = get_fps(mp4_path)
    events = detect_events(y, loaded_sr, hop_length, min_duration_ms, min_peak_db)

    # Add clip-level fields and frame_index
    for e in events:
        e["clip_id"] = clip_id
        e["source_mp4"] = str(mp4_path)
        e["frame_index"] = round(e["timestamp_ms"] / 1000 * fps)

    # Periodicity
    periodicity = compute_periodicity(events)

    # Determine output path
    if output_dir is not None:
        out_path = output_dir / f"{clip_id}_audio_events.jsonl"
    else:
        out_path = mp4_path.parent / f"{clip_id}_audio_events.jsonl"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSONL
    with open(out_path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
        # Summary record
        counts: dict[str, int] = {}
        for e in events:
            counts[e["event_class"]] = counts.get(e["event_class"], 0) + 1
        summary = {
            "clip_id": clip_id,
            "record_type": "summary",
            "n_events": len(events),
            "n_sustained_tone": counts.get("sustained_tone", 0),
            "n_sharp_transient": counts.get("sharp_transient", 0),
            "n_noise_burst": counts.get("noise_burst", 0),
            **{k: v for k, v in periodicity.items() if k != "n_sustained_tone"},
            "sustained_tone_intervals_s": periodicity["sustained_tone_intervals_s"],
            "sustained_tone_mean_interval_s": periodicity["sustained_tone_mean_interval_s"],
            "sustained_tone_cv": periodicity["sustained_tone_cv"],
            "is_periodic": periodicity["is_periodic"],
        }
        f.write(json.dumps(summary) + "\n")

    log.info("Wrote %d events → %s", len(events), out_path)

    if survey:
        print_survey(clip_id, events, periodicity)

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audio landmark detector — extract buzzer/bell events from MP4 clips."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Single MP4 file or directory to scan recursively",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for JSONL files (default: alongside input MP4)",
    )
    parser.add_argument(
        "--survey", action="store_true",
        help="Print human-readable timeline summary to stdout",
    )
    parser.add_argument(
        "--min-duration-ms", type=float, default=200.0,
        help="Minimum event duration in ms to emit (default: 200)",
    )
    parser.add_argument(
        "--min-peak-db", type=float, default=-40.0,
        help="Minimum peak dB to emit (default: -40.0)",
    )
    parser.add_argument(
        "--sr", type=int, default=22050,
        help="Audio sample rate for librosa load (default: 22050)",
    )
    parser.add_argument(
        "--hop-length", type=int, default=512,
        help="STFT hop length (default: 512)",
    )
    args = parser.parse_args()

    mp4s = find_mp4s(args.input)
    if not mp4s:
        log.error("No MP4 files found at: %s", args.input)
        return 1

    log.info("Found %d MP4 file(s)", len(mp4s))

    success = 0
    for mp4 in mp4s:
        if process_clip(mp4, args.output_dir, args.survey, args.min_duration_ms, args.min_peak_db, args.sr, args.hop_length):
            success += 1

    log.info("Done: %d/%d clips processed successfully", success, len(mp4s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
