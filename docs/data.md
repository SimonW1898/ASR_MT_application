[Back to Docs Index](index.md)

# Data Guide

Minimal CVSS + Common Voice workflow for this project.

## Target format

Output root:

- `application/data/raw/cvss_st/<src>-<tgt>/`
  - `clips/`
  - `metadata/train.jsonl`
  - `metadata/validation.jsonl`
  - `metadata/test.jsonl`

Each JSONL row contains:

- `id`
- `src_lang`, `tgt_lang`
- `split`
- `source_audio_relpath`
- `source_text`, `target_text`
- `client_id`, `up_votes`, `down_votes`, `age`, `gender`, `accent`, `segment`

## Data sources
Important for using Data is that the licensing terms of the data are compatible with your intended use. Always review the licenses of the datasets you plan to use. Licenses may have a non commercial clause, which makes them incompatible with commercial use. Only permissible licences are used for the data and models in this repository.
- Source audio + transcript: Mozilla Common Voice [Manual Download](https://commonvoice.mozilla.org/en/datasets) (need to register for access) (License: CC0-1.0)
- Target translation text: [Hugging Face](https://huggingface.co/datasets/google/cvss) `google/cvss` (Older CoVosT version but contains target translations, which is missing for Mozilla Common Voice. License: CC BY 4.0)

Matching key:

- Common Voice TSV: `Path(path).stem`
- CVSS: `sample["id"]`

## Step-by-step (reproducible)

1. Download Common Voice archives manually from Mozilla.
2. Keep archives in `application/` (or extract elsewhere).
3. Run builder (example with tar input):

```powershell
.\venv-eval\Scripts\python.exe .\scripts\data_manage\download_cvss_raw.py --languages ar,de,fa,ja --min_matches 10 --max_per_split 500 --seed 123 --common_voice_tar .\cv-corpus-24.0-2025-12-05-ar.tar.gz
```

If using extracted Common Voice folders instead:

```powershell
.\venv-eval\Scripts\python.exe .\scripts\data_manage\download_cvss_raw.py --languages ar,de,fa,ja --min_matches 10 --max_per_split 500 --seed 123 --common_voice_root <path-to-common-voice-root>
```

Optional checks:

- `--dry_run` computes match rates only and writes nothing.
- `--print_schema` prints the JSONL schema.

## Notes

- Build proceeds per language and only writes data when matched samples >= `--min_matches`.
- At least one language meeting threshold is sufficient for a successful run.
- Clips are hardlinked when possible, with copy fallback.
- Matching takes very long time, because the datasets are very large, so it's recommended to use existing data in zip file or on Campus drive.

[Next: Module Reference](modules.md)
[Back to Docs Index](index.md)
