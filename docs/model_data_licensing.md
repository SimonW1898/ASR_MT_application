[Back to Docs Index](index.md)

# Model/Data Licensing

This page is informational and summarizes where models/data in this repository come from and which licenses apply.

> Not legal advice. Always re-check the upstream model card/dataset card license before distribution.

## Why this matters

- Model and dataset licenses can differ.
- Generated demo artifacts (for example stream JSON/WAV files) usually remain bound by terms of their source data.
- Commercial use, redistribution, attribution, and patent terms depend on the exact license.

## Models used in this project

Source of truth in repo:

- `grpc_demo/configs/model_cache_plan.json`
- `model_cache/hf_snapshots/**/README.md` license fields from cached model cards

### ASR models

- `openai/whisper-small`
  - Upstream: <https://huggingface.co/openai/whisper-small>
  - In cache: `model_cache/hf_snapshots/models--openai--whisper-small/.../README.md`
  - License in cached card: **Apache-2.0**

### MT models

- `Helsinki-NLP/opus-mt-en-de`
  - Upstream: <https://huggingface.co/Helsinki-NLP/opus-mt-en-de>
  - License in cached card: **CC-BY-4.0**
- `Helsinki-NLP/opus-mt-de-en`
  - Upstream: <https://huggingface.co/Helsinki-NLP/opus-mt-de-en>
  - License in cached card: **Apache-2.0**
- `Helsinki-NLP/opus-mt-de-fr`
  - Upstream: <https://huggingface.co/Helsinki-NLP/opus-mt-de-fr>
  - License in cached card: **Apache-2.0**
- `Helsinki-NLP/opus-mt-ar-en`
  - Upstream: <https://huggingface.co/Helsinki-NLP/opus-mt-ar-en>
  - License in cached card: **Apache-2.0**
- `Helsinki-NLP/opus-mt-ja-en`
  - Upstream: <https://huggingface.co/Helsinki-NLP/opus-mt-ja-en>
  - License in cached card: **Apache-2.0**
- `facebook/m2m100_418M`
  - Upstream: <https://huggingface.co/facebook/m2m100_418M>
  - License in cached card: **MIT**

## Data used in this project

Current project data flow is described in `docs/data.md`; this section expands provenance and licensing.

### Primary dataset sources

- Mozilla Common Voice (speech + transcripts)
  - Source: <https://commonvoice.mozilla.org/en/datasets>
  - License used in project docs: **CC0-1.0**
  - Used as the spoken-source side for ST sample construction.
- Google CVSS on Hugging Face (translation-aligned ST rows)
  - Source: <https://huggingface.co/datasets/google/cvss>
  - License used in project docs: **CC BY 4.0**
  - Used to obtain paired translation targets and IDs matched to Common Voice clips.

### Demo stream examples and provenance

- `data/stream/demo_arabic_en.json`
- `data/stream/demo_persian_en.json`
- `data/stream/demo_japanese_en.json`

These JSON files embed segment metadata (`src_wav`, `src_ref`, `tgt_ref`) pointing to files under `data/raw/cvss_st/<pair>/clips`, i.e. derived from the Common Voice + CVSS build path.

Additional legacy sample:

- `data/stream/demo_en_long.json`

This file references `covost2_en_de` paths from an older/legacy local layout and should be treated separately from the current CVSS-based pipeline.

## License summary (plain language)

- **Apache-2.0**
  - Permissive for commercial and non-commercial use.
  - Requires preserving copyright/license notices.
  - Includes an explicit patent license and termination conditions.
- **MIT**
  - Permissive for commercial and non-commercial use.
  - Requires keeping copyright and permission notice.
  - Very short/no-copyleft obligations.
- **CC0-1.0**
  - Public-domain style dedication (where legally possible).
  - Minimal downstream obligations.
- **CC BY 4.0**
  - Allows reuse, modification, and commercial use.
  - Requires attribution and indication of changes.

## Practical guidance for this repository

- Keep attribution metadata when shipping demos derived from CVSS/Common Voice.
- Re-check model card license entries whenever model IDs change in `model_cache_plan.json`.
- If uncertain about redistribution/commercial terms, run a legal review before release.

[Back to Docs Index](index.md)
