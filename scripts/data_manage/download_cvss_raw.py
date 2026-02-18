from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CVSS_RAW_DIR = RAW_DIR / "cvss_st"

SPLITS = ["train", "validation", "test"]
CV_TSV_BY_SPLIT = {
    "train": "train.tsv",
    "validation": "dev.tsv",
    "test": "test.tsv",
}

LANGUAGE_ALIASES = {
    "arabic": "ar",
    "german": "de",
    "japanese": "ja",
    "russian": "ru",
    "farsi": "fa",
}


def _log(message: str) -> None:
    print(f"[cvss_raw_builder] {message}")


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _to_optional_int(value: str | None) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _normalize_language(value: str) -> str:
    key = value.strip().lower().replace("_", "-")
    return LANGUAGE_ALIASES.get(key, key)


def _resolve_text(sample: dict[str, Any]) -> str:
    for key in ("text", "translation", "target_text", "normalized_text"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


@dataclass(frozen=True)
class CommonVoiceRecord:
    key: str
    filename: str
    source_text: str
    client_id: str | None
    up_votes: int | None
    down_votes: int | None
    age: str | None
    gender: str | None
    accent: str | None
    segment: str | None
    source_audio_path: Path | None
    tar_member_name: str | None


@dataclass(frozen=True)
class CVSSRecord:
    key: str
    target_text: str
    split: str


@dataclass(frozen=True)
class MergeStats:
    language: str
    cvss_total: int
    matched: int
    written: int
    match_rate: float


class CommonVoiceIndex:
    def __init__(
        self,
        language: str,
        common_voice_root: Path | None,
        common_voice_tar: Path | None,
    ) -> None:
        self.language = language
        self.common_voice_root = common_voice_root
        self.common_voice_tar = common_voice_tar
        self.records_by_split: dict[str, dict[str, CommonVoiceRecord]] = {
            split: {} for split in SPLITS
        }
        self._tar_prefix: str | None = None

    def build(self) -> None:
        if self.common_voice_root is None and self.common_voice_tar is None:
            raise RuntimeError("Provide --common_voice_root or --common_voice_tar.")
        if self.common_voice_root is not None:
            self._build_from_root()
        else:
            self._build_from_tar()

        total_rows = sum(len(rows) for rows in self.records_by_split.values())
        if total_rows == 0:
            raise RuntimeError(
                f"No Common Voice TSV rows were indexed for language '{self.language}'."
            )

    def _build_from_root(self) -> None:
        assert self.common_voice_root is not None
        lang_dir = self._resolve_language_dir(self.common_voice_root, self.language)
        clips_dir = lang_dir / "clips"
        if not clips_dir.is_dir():
            raise RuntimeError(f"No clips directory found: {clips_dir}")

        tsv_found = False
        for split in SPLITS:
            tsv_path = lang_dir / CV_TSV_BY_SPLIT[split]
            if not tsv_path.exists():
                continue
            tsv_found = True
            with tsv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    rel_path = str(row.get("path", "") or "").strip()
                    if not rel_path:
                        continue
                    key = Path(rel_path).stem
                    if not key:
                        continue
                    record = CommonVoiceRecord(
                        key=key,
                        filename=Path(rel_path).name,
                        source_text=str(row.get("sentence", "") or "").strip(),
                        client_id=str(row.get("client_id", "") or "").strip() or None,
                        up_votes=_to_optional_int(row.get("up_votes")),
                        down_votes=_to_optional_int(row.get("down_votes")),
                        age=str(row.get("age", "") or "").strip() or None,
                        gender=str(row.get("gender", "") or "").strip() or None,
                        accent=(
                            str(row.get("accents", "") or row.get("accent", "")).strip() or None
                        ),
                        segment=str(row.get("segment", "") or "").strip() or None,
                        source_audio_path=(clips_dir / rel_path),
                        tar_member_name=None,
                    )
                    self.records_by_split[split][key] = record

        if not tsv_found:
            raise RuntimeError(
                f"No Common Voice TSV found under {lang_dir} (expected train.tsv/dev.tsv/test.tsv)."
            )

    def _build_from_tar(self) -> None:
        assert self.common_voice_tar is not None
        if not self.common_voice_tar.exists():
            raise RuntimeError(f"Common Voice tar not found: {self.common_voice_tar}")

        with tarfile.open(self.common_voice_tar, mode="r:gz") as tar:
            members = tar.getmembers()
            by_name = {m.name: m for m in members}
            tsv_found = False

            for split in SPLITS:
                suffix = f"/{self.language}/{CV_TSV_BY_SPLIT[split]}"
                tsv_member = next((m for m in members if m.name.endswith(suffix)), None)
                if tsv_member is None:
                    continue
                tsv_found = True
                marker = f"/{self.language}/"
                idx = tsv_member.name.find(marker)
                if idx <= 0:
                    continue
                self._tar_prefix = tsv_member.name[:idx]

                stream = tar.extractfile(tsv_member)
                if stream is None:
                    continue

                text_lines = stream.read().decode("utf-8", errors="replace").splitlines()
                reader = csv.DictReader(text_lines, delimiter="\t")
                for row in reader:
                    rel_path = str(row.get("path", "") or "").strip()
                    if not rel_path:
                        continue
                    key = Path(rel_path).stem
                    if not key:
                        continue
                    member_name = f"{self._tar_prefix}/{self.language}/clips/{rel_path}"
                    if member_name not in by_name:
                        continue
                    record = CommonVoiceRecord(
                        key=key,
                        filename=Path(rel_path).name,
                        source_text=str(row.get("sentence", "") or "").strip(),
                        client_id=str(row.get("client_id", "") or "").strip() or None,
                        up_votes=_to_optional_int(row.get("up_votes")),
                        down_votes=_to_optional_int(row.get("down_votes")),
                        age=str(row.get("age", "") or "").strip() or None,
                        gender=str(row.get("gender", "") or "").strip() or None,
                        accent=(
                            str(row.get("accents", "") or row.get("accent", "")).strip() or None
                        ),
                        segment=str(row.get("segment", "") or "").strip() or None,
                        source_audio_path=None,
                        tar_member_name=member_name,
                    )
                    self.records_by_split[split][key] = record

            if not tsv_found:
                raise RuntimeError(
                    f"No Common Voice TSV found in tar for language '{self.language}'."
                )

    @staticmethod
    def _resolve_language_dir(root: Path, language: str) -> Path:
        candidates = [
            root / language,
            root,
        ]
        for candidate in candidates:
            if (candidate / "clips").is_dir():
                return candidate
        nested = sorted(root.glob(f"cv-corpus-*/{language}"))
        for candidate in nested:
            if (candidate / "clips").is_dir():
                return candidate
        raise RuntimeError(f"No Common Voice language directory found for '{language}' under {root}")

    def read_audio_bytes(self, record: CommonVoiceRecord) -> bytes:
        if record.source_audio_path is not None:
            if not record.source_audio_path.exists():
                raise RuntimeError(f"Missing clip file: {record.source_audio_path}")
            return record.source_audio_path.read_bytes()
        if self.common_voice_tar is None or not record.tar_member_name:
            raise RuntimeError("Record has no resolvable audio source.")

        with tarfile.open(self.common_voice_tar, mode="r:gz") as tar:
            stream = tar.extractfile(record.tar_member_name)
            if stream is None:
                raise RuntimeError(f"Missing tar member: {record.tar_member_name}")
            return stream.read()


class CVSSLoader:
    def __init__(self, language: str) -> None:
        self.language = language

    def load(self) -> tuple[dict[str, list[CVSSRecord]], int]:
        _log(f"load_cvss language={self.language} config=cvss_c")
        dataset = load_dataset(
            "google/cvss",
            "cvss_c",
            languages=[self.language],
            trust_remote_code=True,
        )

        if not isinstance(dataset, DatasetDict):
            raise RuntimeError("Expected google/cvss loader to return DatasetDict.")

        records_by_split: dict[str, list[CVSSRecord]] = {split: [] for split in SPLITS}
        cvss_total = 0
        for split in SPLITS:
            ds_split = self._resolve_split(dataset, split)
            if ds_split is None:
                continue
            for sample in ds_split:
                raw_id = str(sample.get("id", "") or "").strip()
                key = Path(raw_id).stem if raw_id else ""
                target_text = _resolve_text(sample)
                if not key or not target_text:
                    continue
                records_by_split[split].append(
                    CVSSRecord(
                        key=key,
                        target_text=target_text,
                        split=split,
                    )
                )
                cvss_total += 1
        return records_by_split, cvss_total

    @staticmethod
    def _resolve_split(dataset: DatasetDict, split: str):
        if split in dataset:
            return dataset[split]
        if split == "validation" and "dev" in dataset:
            return dataset["dev"]
        return None


class DatasetMerger:
    def __init__(
        self,
        language: str,
        common_voice_index: CommonVoiceIndex,
        cvss_by_split: dict[str, list[CVSSRecord]],
        cvss_total: int,
        output_root: Path,
        min_matches: int,
        max_per_split: int,
        seed: int,
        dry_run: bool,
    ) -> None:
        self.language = language
        self.target_lang = "en"
        self.common_voice_index = common_voice_index
        self.cvss_by_split = cvss_by_split
        self.cvss_total = cvss_total
        self.output_root = output_root
        self.min_matches = min_matches
        self.max_per_split = max_per_split
        self.seed = seed
        self.dry_run = dry_run

    def run(self) -> MergeStats:
        pair = f"{self.language}-{self.target_lang}"
        pair_dir = self.output_root / pair
        clips_dir = pair_dir / "clips"
        metadata_dir = pair_dir / "metadata"

        matched_by_split: dict[str, list[tuple[CVSSRecord, CommonVoiceRecord]]] = {
            split: [] for split in SPLITS
        }
        for split in SPLITS:
            cv_map = self.common_voice_index.records_by_split.get(split, {})
            for cvss_row in self.cvss_by_split.get(split, []):
                cv_row = cv_map.get(cvss_row.key)
                if cv_row is None:
                    continue
                matched_by_split[split].append((cvss_row, cv_row))

        matched = sum(len(rows) for rows in matched_by_split.values())
        match_rate = (matched / self.cvss_total * 100.0) if self.cvss_total > 0 else 0.0

        if matched < self.min_matches:
            raise RuntimeError(
                f"match_rate below threshold for {pair}: matched={matched}, "
                f"cvss_total={self.cvss_total}, match_rate={match_rate:.2f}%, min_matches={self.min_matches}"
            )

        if self.dry_run:
            return MergeStats(
                language=pair,
                cvss_total=self.cvss_total,
                matched=matched,
                written=0,
                match_rate=match_rate,
            )

        clips_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        total_written = 0
        for split_index, split in enumerate(SPLITS):
            split_rows = matched_by_split.get(split, [])
            rng = random.Random(self.seed + split_index)
            rng.shuffle(split_rows)
            selected = split_rows[: self.max_per_split]

            output_rows: list[dict[str, Any]] = []
            for row_index, (cvss_row, cv_row) in enumerate(selected):
                file_ext = Path(cv_row.filename).suffix or ".mp3"
                clip_name = f"{split}_{cvss_row.key}_{row_index:06d}{file_ext}"
                clip_path = clips_dir / clip_name

                self._materialize_clip(cv_row, clip_path)

                output_rows.append(
                    {
                        "id": cvss_row.key,
                        "src_lang": self.language,
                        "tgt_lang": self.target_lang,
                        "split": split,
                        "source_audio_relpath": str(Path("clips") / clip_name).replace("\\", "/"),
                        "source_text": cv_row.source_text,
                        "target_text": cvss_row.target_text,
                        "client_id": cv_row.client_id,
                        "up_votes": cv_row.up_votes,
                        "down_votes": cv_row.down_votes,
                        "age": cv_row.age,
                        "gender": cv_row.gender,
                        "accent": cv_row.accent,
                        "segment": cv_row.segment,
                    }
                )

            out_jsonl = metadata_dir / f"{split}.jsonl"
            with out_jsonl.open("w", encoding="utf-8") as f:
                for item in output_rows:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            self._validate_written_rows(pair_dir, output_rows)
            total_written += len(output_rows)

        return MergeStats(
            language=pair,
            cvss_total=self.cvss_total,
            matched=matched,
            written=total_written,
            match_rate=match_rate,
        )

    def _materialize_clip(self, record: CommonVoiceRecord, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            return

        if record.source_audio_path is not None and record.source_audio_path.exists():
            try:
                destination.hardlink_to(record.source_audio_path)
                return
            except Exception:
                shutil.copy2(record.source_audio_path, destination)
                return

        blob = self.common_voice_index.read_audio_bytes(record)
        with destination.open("wb") as f:
            f.write(blob)

    @staticmethod
    def _validate_written_rows(pair_dir: Path, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            rel = str(row.get("source_audio_relpath", "") or "").strip()
            if not rel:
                raise RuntimeError("Missing source_audio_relpath in written record")
            target = pair_dir / rel
            if not target.exists():
                raise RuntimeError(f"Written metadata points to missing clip: {target}")


def _print_schema() -> None:
    schema = {
        "id": "<key>",
        "src_lang": "<src>",
        "tgt_lang": "en",
        "split": "train|validation|test",
        "source_audio_relpath": "clips/<filename>",
        "source_text": "<CV transcript>",
        "target_text": "<CVSS translation>",
        "client_id": "<str|null>",
        "up_votes": "<int|null>",
        "down_votes": "<int|null>",
        "age": "<str|null>",
        "gender": "<str|null>",
        "accent": "<str|null>",
        "segment": "<str|null>",
    }
    print(json.dumps(schema, ensure_ascii=False, indent=2))


def _print_summary(stats: list[MergeStats]) -> None:
    print("Language | CVSS total | Matched | Written | Match rate %")
    for row in stats:
        print(
            f"{row.language} | {row.cvss_total} | {row.matched} | {row.written} | {row.match_rate:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CVSS+CommonVoice permissive raw dataset.")
    parser.add_argument("--languages", default="ar,de,fa,ja", help="CSV source languages")
    parser.add_argument("--min_matches", type=int, default=10, help="Minimum matched samples per language")
    parser.add_argument("--max_per_split", type=int, default=1000, help="Maximum rows to write per split")
    parser.add_argument("--seed", type=int, default=123, help="Deterministic sampling seed")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Compute matches only, no writes")
    parser.add_argument("--print_schema", action="store_true", default=False, help="Print output JSONL schema")
    parser.add_argument("--common_voice_root", default=None, help="Path to extracted Common Voice root")
    parser.add_argument("--common_voice_tar", default=None, help="Path to Common Voice tar.gz archive")
    args = parser.parse_args()

    if args.print_schema:
        _print_schema()

    if args.common_voice_root is None and args.common_voice_tar is None:
        raise RuntimeError("Either --common_voice_root or --common_voice_tar must be provided.")

    common_voice_root = Path(args.common_voice_root).resolve() if args.common_voice_root else None
    common_voice_tar = Path(args.common_voice_tar).resolve() if args.common_voice_tar else None

    languages = [_normalize_language(item) for item in _parse_csv(args.languages)]
    languages = [item for item in languages if item]
    if not languages:
        raise RuntimeError("No languages provided.")

    output_root = CVSS_RAW_DIR
    output_root.mkdir(parents=True, exist_ok=True)

    stats: list[MergeStats] = []
    written_any = False
    for language in languages:
        _log(f"language_start src={language} tgt=en")
        cv_index = CommonVoiceIndex(
            language=language,
            common_voice_root=common_voice_root,
            common_voice_tar=common_voice_tar,
        )
        cv_index.build()

        cvss_loader = CVSSLoader(language=language)
        cvss_by_split, cvss_total = cvss_loader.load()

        merger = DatasetMerger(
            language=language,
            common_voice_index=cv_index,
            cvss_by_split=cvss_by_split,
            cvss_total=cvss_total,
            output_root=output_root,
            min_matches=args.min_matches,
            max_per_split=args.max_per_split,
            seed=args.seed,
            dry_run=args.dry_run,
        )

        lang_stats = merger.run()
        stats.append(lang_stats)
        if lang_stats.written > 0:
            written_any = True
        _log(
            f"language_done src={language} cvss_total={lang_stats.cvss_total} "
            f"matched={lang_stats.matched} written={lang_stats.written} rate={lang_stats.match_rate:.2f}%"
        )

    if not stats:
        raise RuntimeError("No languages processed.")

    _print_summary(stats)

    if not args.dry_run and not written_any:
        raise RuntimeError(
            "No language produced written output. Match rate < min_matches for all languages."
        )


if __name__ == "__main__":
    main()
