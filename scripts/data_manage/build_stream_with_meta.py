from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


SCRIPT_PATH = Path(__file__).resolve()
APPLICATION_DIR = SCRIPT_PATH.parents[2]
DATA_DIR = APPLICATION_DIR / "data"
CVSS_RAW_DIR = DATA_DIR / "raw" / "cvss_st"
STREAM_DIR = DATA_DIR / "stream"
TARGET_SAMPLE_RATE = 16_000


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _stable_dir_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()


def _to_mono(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if arr.shape[0] <= 4 and arr.shape[1] > 4:
            return arr.mean(axis=0, dtype=np.float32)
        return arr.mean(axis=1, dtype=np.float32)
    return arr.reshape(-1).astype(np.float32)


def _resample_linear(audio: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    if src_sr == tgt_sr:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)

    ratio = tgt_sr / src_sr
    target_len = max(1, int(round(audio.shape[0] * ratio)))
    x_old = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _normalize_quotes(text: str) -> str:
    if not text:
        return text
    text = (
        text.replace("„", '"')
        .replace("“", '"')
        .replace("”", '"')
        .replace("«", '"')
        .replace("»", '"')
        .replace("’", "'")
    )
    text = text.replace('"', "")
    text = text.replace("''", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * TARGET_SAMPLE_RATE), dtype=np.float32)


def build_stream_with_meta(
    metadata_rows: list[dict[str, Any]],
    pair_dir: Path,
    *,
    sentences_per_stream: int,
    min_gap_s: float,
    max_gap_s: float,
    seed: int,
    target_duration_s: float | None = None,
    shuffle_rows: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = random.Random(seed)
    rows = list(metadata_rows)
    if shuffle_rows:
        rng.shuffle(rows)

    chunks: list[np.ndarray] = []
    segments: list[dict[str, Any]] = []
    stream_pos_samples = 0

    def add_segment(kind: str, audio: np.ndarray, row: dict[str, Any] | None = None, src_wav: Path | None = None) -> None:
        nonlocal stream_pos_samples
        n = int(audio.shape[0])
        start = stream_pos_samples
        end = start + n

        segment: dict[str, Any] = {
            "kind": kind,
            "stream_start_sample": start,
            "stream_end_sample": end,
            "stream_start_s": start / TARGET_SAMPLE_RATE,
            "stream_end_s": end / TARGET_SAMPLE_RATE,
            "duration_s": n / TARGET_SAMPLE_RATE,
        }

        if row is not None:
            segment["src_wav"] = str(src_wav.resolve()) if src_wav is not None else ""
            segment["src_wav_name"] = Path(str(row.get("source_audio_relpath", ""))).name
            segment["src_ref"] = _normalize_quotes(str(row.get("source_text", "")))
            segment["tgt_ref"] = _normalize_quotes(str(row.get("target_text", "")))

        chunks.append(audio)
        segments.append(segment)
        stream_pos_samples = end

    used_utt = 0
    for row in rows:
        rel_audio = str(row.get("source_audio_relpath", "")).strip()
        if not rel_audio:
            continue

        wav_path = pair_dir / rel_audio
        if not wav_path.exists():
            continue

        clip, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        clip = _to_mono(clip)
        clip = _resample_linear(clip, src_sr=int(sr), tgt_sr=TARGET_SAMPLE_RATE)
        if clip.size == 0:
            continue

        add_segment(kind="utt", audio=clip, row=row, src_wav=wav_path)
        used_utt += 1
        if used_utt >= sentences_per_stream:
            break

        gap = rng.uniform(min_gap_s, max_gap_s)
        if gap > 0:
            add_segment(kind="silence", audio=_silence(gap), row=None)

    stream_audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    final_duration_s = float(len(stream_audio) / TARGET_SAMPLE_RATE)

    meta = {
        "schema": "long_stream_v1",
        "sr": TARGET_SAMPLE_RATE,
        "dtype": "float32",
        "seed": seed,
        "target_duration_s": float(target_duration_s if target_duration_s is not None else final_duration_s),
        "min_gap_s": float(min_gap_s),
        "max_gap_s": float(max_gap_s),
        "actual_duration_s": final_duration_s,
        "num_segments": len(segments),
        "num_utt_segments": sum(1 for s in segments if s["kind"] == "utt"),
        "num_silence_segments": sum(1 for s in segments if s["kind"] == "silence"),
        "input_files_dir_id": _stable_dir_id(pair_dir / "clips"),
        "segments": segments,
    }

    return stream_audio, meta


def _load_pair_rows(pair_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metadata_dir = pair_dir / "metadata"
    for split in ("train", "validation", "test"):
        split_path = metadata_dir / f"{split}.jsonl"
        if not split_path.exists():
            continue
        rows.extend(_read_jsonl(split_path))
    return rows


def _chunk_rows(rows: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    if chunk_size <= 0:
        return []
    return [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build long-stream WAV + metadata from CVSS raw data."
    )
    parser.add_argument(
        "--pairs",
        required=True,
        help="CSV language pairs (example: fa-en,ja-en).",
    )
    parser.add_argument("--num-streams", type=int, default=1, help="Streams per pair.")
    parser.add_argument(
        "--sentences-per-stream",
        type=int,
        default=5,
        help="Number of utterances concatenated into each stream.",
    )
    parser.add_argument(
        "--target-duration",
        type=float,
        default=None,
        help="Optional metadata-only target duration value.",
    )
    parser.add_argument("--min-gap", type=float, default=0.2, help="Minimum silence gap seconds.")
    parser.add_argument("--max-gap", type=float, default=0.6, help="Maximum silence gap seconds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed base.")
    parser.add_argument(
        "--use-all-sentences",
        action="store_true",
        default=False,
        help="Use each available sentence at most once by chunking rows into fixed-size streams.",
    )
    parser.add_argument(
        "--max-samples-per-pair",
        type=int,
        default=None,
        help="Optional cap on metadata rows consumed per pair.",
    )
    args = parser.parse_args()

    pairs = _parse_csv(args.pairs)
    if not pairs:
        raise ValueError("No valid pairs provided.")

    if args.sentences_per_stream < 1:
        raise ValueError("--sentences-per-stream must be >= 1")

    STREAM_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[build_stream_with_meta] application_dir={APPLICATION_DIR}")
    print(f"[build_stream_with_meta] data_dir={DATA_DIR}")

    for pair in pairs:
        pair_raw_dir = CVSS_RAW_DIR / pair
        rows = _load_pair_rows(pair_raw_dir)
        if args.max_samples_per_pair is not None:
            rows = rows[: args.max_samples_per_pair]

        if not rows:
            print(f"[build_stream_with_meta] skip pair={pair} (no metadata rows)")
            continue

        pair_output = STREAM_DIR / pair
        pair_output.mkdir(parents=True, exist_ok=True)

        if args.use_all_sentences:
            pair_rng = random.Random(args.seed)
            shuffled_rows = list(rows)
            pair_rng.shuffle(shuffled_rows)
            row_groups = _chunk_rows(shuffled_rows, args.sentences_per_stream)
            row_groups = [group for group in row_groups if len(group) == args.sentences_per_stream]
        else:
            row_groups = [rows for _ in range(args.num_streams)]

        for i, row_group in enumerate(row_groups):
            seed_i = args.seed + i
            stream_audio, meta = build_stream_with_meta(
                metadata_rows=row_group,
                pair_dir=pair_raw_dir,
                sentences_per_stream=args.sentences_per_stream,
                min_gap_s=args.min_gap,
                max_gap_s=args.max_gap,
                seed=seed_i,
                target_duration_s=args.target_duration,
                shuffle_rows=not args.use_all_sentences,
            )

            base = f"stream_{i:02d}_{args.sentences_per_stream}utt"
            out_wav = pair_output / f"{base}.wav"
            out_json = pair_output / f"{base}.json"

            sf.write(out_wav, stream_audio, TARGET_SAMPLE_RATE, subtype="PCM_16")

            meta["stream_wav"] = str(out_wav.resolve())
            meta["stream_json"] = str(out_json.resolve())
            with out_json.open("w", encoding="utf-8") as handle:
                json.dump(meta, handle, ensure_ascii=False, indent=2)

            print(
                f"[build_stream_with_meta] pair={pair} stream={i} "
                f"duration_s={meta['actual_duration_s']:.2f} utt={meta['num_utt_segments']} "
                f"sil={meta['num_silence_segments']}"
            )


if __name__ == "__main__":
    main()
