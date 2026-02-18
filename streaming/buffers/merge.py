"""
Transcript merge utilities for overlap-style streaming ASR.
"""

from dataclasses import dataclass
from typing import List, Tuple
import difflib
import re


@dataclass(frozen=True)
class MergeConfig:
    """
    Configuration for overlap-based transcript merging.

    n_words: how many words from the end of the previous transcript to consider.
    min_match: minimum consecutive word match length to accept an overlap.
    lowercase: normalize case before matching.
    strip_punct: optionally strip punctuation before matching.
    """
    n_words: int = 7
    min_match: int = 2
    lowercase: bool = True
    strip_punct: bool = False


_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def _normalize_text(text: str, *, lowercase: bool, strip_punct: bool) -> str:
    text = text.strip()
    text = _WS_RE.sub(" ", text)
    if lowercase:
        text = text.lower()
    if strip_punct:
        text = _PUNCT_RE.sub("", text)
        text = _WS_RE.sub(" ", text).strip()
    return text


def _tokenize(text: str, *, lowercase: bool, strip_punct: bool) -> List[str]:
    norm = _normalize_text(text, lowercase=lowercase, strip_punct=strip_punct)
    return norm.split() if norm else []


def merge_transcripts(prev_text: str, new_text: str, cfg: MergeConfig = MergeConfig()) -> str:
    """
    Merge two ASR hypotheses that may overlap due to audio context reuse.
    """
    if not prev_text:
        return new_text
    if not new_text:
        return prev_text

    prev_words = _tokenize(prev_text, lowercase=cfg.lowercase, strip_punct=cfg.strip_punct)
    new_words = _tokenize(new_text, lowercase=cfg.lowercase, strip_punct=cfg.strip_punct)

    if not prev_words:
        return new_text
    if not new_words:
        return prev_text

    tail = prev_words[-cfg.n_words:] if len(prev_words) > cfg.n_words else prev_words

    best = None  # (match_len, prev_keep_upto_idx, new_start_idx)
    for match_len in range(len(tail), cfg.min_match - 1, -1):
        tail_suffix = tail[-match_len:]
        for i in range(0, len(new_words) - match_len + 1):
            if new_words[i:i + match_len] == tail_suffix:
                prev_keep_upto = len(prev_words) - match_len
                best = (match_len, prev_keep_upto, i)
                break
        if best is not None:
            break

    if best is None:
        return (prev_text.rstrip() + " " + new_text.lstrip()).strip()

    _, prev_keep_upto, new_start = best
    merged_words = prev_words[:prev_keep_upto] + new_words[new_start:]
    return " ".join(merged_words)


@dataclass(frozen=True)
class FuzzyMergeConfig:
    window_chars: int = 320
    min_overlap_chars: int = 2
    sim_threshold: float = 0.88

    lowercase: bool = True
    normalize_ws: bool = True
    normalize_quotes: bool = True

    enable_seam_revision: bool = True
    seam_chars: int = 80
    seam_sim_threshold: float = 0.90


_WS_RE = re.compile(r"\s+", flags=re.UNICODE)
_LEADING_JUNK_RE = re.compile(r"^[\s\.\,\;\:\!\?\)\]\}]+", flags=re.UNICODE)


def _normalize_for_match(s: str, cfg: FuzzyMergeConfig) -> str:
    s = s.strip()

    if cfg.normalize_quotes:
        s = (
            s.replace("\u2019", "'").replace("\u2018", "'")
            .replace("\u201c", '"').replace("\u201d", '"')
        )

    if cfg.lowercase:
        s = s.lower()

    if cfg.normalize_ws:
        s = _WS_RE.sub(" ", s)

    return s


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    d = _levenshtein_distance(a, b)
    denom = max(len(a), len(b), 1)
    return 1.0 - (d / denom)


def _best_overlap_norm(prev_text: str, new_text: str, cfg: FuzzyMergeConfig) -> Tuple[int, float]:
    prev_n = _normalize_for_match(prev_text, cfg)
    new_n = _normalize_for_match(new_text, cfg)
    if not prev_n or not new_n:
        return 0, 0.0

    prev_w = prev_n[-cfg.window_chars:] if len(prev_n) > cfg.window_chars else prev_n
    new_w = new_n[:cfg.window_chars] if len(new_n) > cfg.window_chars else new_n

    max_L = min(len(prev_w), len(new_w))
    if max_L < cfg.min_overlap_chars:
        return 0, 0.0

    best_sim = 0.0

    for L in range(max_L, cfg.min_overlap_chars - 1, -1):
        a = prev_w[-L:]
        b = new_w[:L]
        sim = _similarity(a, b)
        if sim >= cfg.sim_threshold:
            return L, sim
        if sim > best_sim:
            best_sim = sim

    return 0, best_sim


def _normalized_prefix_len_to_raw_cut(raw_new: str, cfg: FuzzyMergeConfig, L_norm: int) -> int:
    if L_norm <= 0:
        return 0

    norm_count = 0
    last_norm_char = None

    for raw_i, ch in enumerate(raw_new):
        if cfg.normalize_quotes:
            if ch in ("\u2019", "\u2018"):
                ch2 = "'"
            elif ch in ("\u201c", "\u201d"):
                ch2 = '"'
            else:
                ch2 = ch
        else:
            ch2 = ch

        if cfg.lowercase:
            ch2 = ch2.lower()

        if cfg.normalize_ws and ch2.isspace():
            ch2 = " "
            if last_norm_char == " ":
                continue

        norm_count += 1
        last_norm_char = ch2

        if norm_count >= L_norm:
            return raw_i + 1

    return 0


def _snap_cut_to_boundary(raw_new: str, cut: int) -> int:
    cut = max(0, min(cut, len(raw_new)))
    if cut <= 0 or cut >= len(raw_new):
        return cut

    if raw_new[cut - 1].isalnum() and raw_new[cut].isalnum():
        while cut < len(raw_new) and raw_new[cut].isalnum():
            cut += 1

    if cut < len(raw_new) and raw_new[cut] == "'" and cut > 0 and raw_new[cut - 1].isalnum():
        cut += 1
        while cut < len(raw_new) and raw_new[cut].isalpha():
            cut += 1

    return cut


def _clean_remainder(remainder: str) -> str:
    return _LEADING_JUNK_RE.sub("", remainder).lstrip()


def _maybe_seam_revision(prev_text: str, new_text: str, cfg: FuzzyMergeConfig) -> str:
    if not cfg.enable_seam_revision:
        return prev_text
    if not prev_text or not new_text:
        return prev_text

    prev_tail = prev_text[-cfg.seam_chars:] if len(prev_text) > cfg.seam_chars else prev_text
    new_head = new_text[:cfg.seam_chars] if len(new_text) > cfg.seam_chars else new_text

    prev_n = _normalize_for_match(prev_tail, cfg)
    new_n = _normalize_for_match(new_head, cfg)
    if not prev_n or not new_n:
        return prev_text

    sim = _similarity(prev_n, new_n)
    if sim < cfg.seam_sim_threshold:
        return prev_text

    return (prev_text[:-len(prev_tail)] + new_head).rstrip()


def merge_transcripts_fuzzy(prev_text: str, new_text: str, cfg: FuzzyMergeConfig = FuzzyMergeConfig()) -> str:
    if not prev_text:
        return new_text
    if not new_text:
        return prev_text

    prev_text = _maybe_seam_revision(prev_text, new_text, cfg)

    L_norm, _sim = _best_overlap_norm(prev_text, new_text, cfg)
    if L_norm <= 0:
        return (prev_text.rstrip() + " " + new_text.lstrip()).strip()

    cut = _normalized_prefix_len_to_raw_cut(new_text, cfg, L_norm)
    if cut <= 0 or cut > len(new_text):
        return (prev_text.rstrip() + " " + new_text.lstrip()).strip()

    cut = _snap_cut_to_boundary(new_text, cut)
    rest = _clean_remainder(new_text[cut:])

    if not rest.strip():
        return prev_text.strip()

    return (prev_text.rstrip() + " " + rest).strip()


def merge_transcripts_fuzzy_with_info(
    prev_text: str,
    new_text: str,
    cfg: FuzzyMergeConfig = FuzzyMergeConfig(),
) -> Tuple[str, dict]:
    if not prev_text:
        return new_text, {"mode": "init", "overlap_norm_chars": 0, "sim": 1.0, "cut_raw_chars": 0, "seam_revised": False}
    if not new_text:
        return prev_text, {"mode": "noop", "overlap_norm_chars": 0, "sim": 1.0, "cut_raw_chars": 0, "seam_revised": False}

    seam_revised = False
    prev2 = prev_text
    prev_text = _maybe_seam_revision(prev_text, new_text, cfg)
    if prev_text != prev2:
        seam_revised = True

    L_norm, best_sim = _best_overlap_norm(prev_text, new_text, cfg)
    if L_norm <= 0:
        merged = (prev_text.rstrip() + " " + new_text.lstrip()).strip()
        return merged, {"mode": "fallback", "overlap_norm_chars": 0, "sim": best_sim, "cut_raw_chars": 0, "seam_revised": seam_revised}

    cut = _normalized_prefix_len_to_raw_cut(new_text, cfg, L_norm)
    if cut <= 0 or cut > len(new_text):
        merged = (prev_text.rstrip() + " " + new_text.lstrip()).strip()
        return merged, {"mode": "fallback_cut", "overlap_norm_chars": L_norm, "sim": best_sim, "cut_raw_chars": 0, "seam_revised": seam_revised}

    cut2 = _snap_cut_to_boundary(new_text, cut)
    rest = _clean_remainder(new_text[cut2:])
    if not rest.strip():
        return prev_text.strip(), {"mode": "match_empty", "overlap_norm_chars": L_norm, "sim": best_sim, "cut_raw_chars": cut2, "seam_revised": seam_revised}

    merged = (prev_text.rstrip() + " " + rest).strip()
    return merged, {"mode": "match", "overlap_norm_chars": L_norm, "sim": best_sim, "cut_raw_chars": cut2, "seam_revised": seam_revised}


# Alignment-based merge

_TOKEN_RE = re.compile(r"\w+(?:'\w+)?|[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)
_TRAILING_PUNCT_RE = re.compile(r"[\.,;:!?\)\]\}]+$", flags=re.UNICODE)


def _norm_token(tok: str) -> str:
    t = tok.lower()
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    t = _TRAILING_PUNCT_RE.sub("", t)
    return t


def _tokenize_align(text: str) -> List[str]:
    text = _WS_RE.sub(" ", text.strip())
    return _TOKEN_RE.findall(text) if text else []


def _detokenize(tokens: List[str]) -> str:
    out = []
    for i, tok in enumerate(tokens):
        if i == 0:
            out.append(tok)
            continue
        if re.fullmatch(r"[,\.\!\?\:\;\)\]\}]", tok):
            out.append(tok)
        elif re.fullmatch(r"[\(\[\{]", tok):
            out.append(" " + tok)
        else:
            out.append(" " + tok)
    return "".join(out).strip()


@dataclass(frozen=True)
class AlignMergeConfig:
    tail_tokens: int = 50
    head_tokens: int = 50
    min_match_tokens: int = 1
    max_search_shift: int = 8


def merge_transcripts_align(prev_text: str, new_text: str, cfg: AlignMergeConfig = AlignMergeConfig()) -> str:
    if not prev_text:
        return new_text
    if not new_text:
        return prev_text

    prev_toks = _tokenize_align(prev_text)
    new_toks = _tokenize_align(new_text)
    if not prev_toks:
        return new_text
    if not new_toks:
        return prev_text

    prev_tail = prev_toks[-cfg.tail_tokens:] if len(prev_toks) > cfg.tail_tokens else prev_toks
    new_head = new_toks[:cfg.head_tokens] if len(new_toks) > cfg.head_tokens else new_toks

    a = [_norm_token(t) for t in prev_tail]
    b = [_norm_token(t) for t in new_head]

    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)

    best = None  # (match_len, a_start, b_start)
    for (a0, b0, n) in sm.get_matching_blocks():
        if n <= 0:
            continue
        if n < cfg.min_match_tokens:
            continue
        a_end = a0 + n
        if (len(a) - a_end) > cfg.max_search_shift:
            continue
        if best is None or n > best[0]:
            best = (n, a0, b0)

    if best is None:
        return (prev_text.rstrip() + " " + new_text.lstrip()).strip()

    n, a0, b0 = best
    prev_keep = len(prev_toks) - len(prev_tail) + a0
    merged = prev_toks[:prev_keep] + new_toks[b0:]

    return _detokenize(merged)


def merge_transcripts_align_with_info(
    prev_text: str,
    new_text: str,
    cfg: AlignMergeConfig = AlignMergeConfig(),
) -> Tuple[str, dict]:
    if not prev_text:
        return new_text, {"mode": "init", "match_len": 0, "prev_keep": 0, "new_start": 0}
    if not new_text:
        return prev_text, {"mode": "noop", "match_len": 0, "prev_keep": 0, "new_start": 0}

    prev_toks = _tokenize_align(prev_text)
    new_toks = _tokenize_align(new_text)
    if not prev_toks:
        return new_text, {"mode": "init", "match_len": 0, "prev_keep": 0, "new_start": 0}
    if not new_toks:
        return prev_text, {"mode": "noop", "match_len": 0, "prev_keep": 0, "new_start": 0}

    prev_tail = prev_toks[-cfg.tail_tokens:] if len(prev_toks) > cfg.tail_tokens else prev_toks
    new_head = new_toks[:cfg.head_tokens] if len(new_toks) > cfg.head_tokens else new_toks

    a = [_norm_token(t) for t in prev_tail]
    b = [_norm_token(t) for t in new_head]

    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)

    best = None
    for (a0, b0, n) in sm.get_matching_blocks():
        if n <= 0:
            continue
        if n < cfg.min_match_tokens:
            continue
        a_end = a0 + n
        if (len(a) - a_end) > cfg.max_search_shift:
            continue
        if best is None or n > best[0]:
            best = (n, a0, b0)

    if best is None:
        merged = (prev_text.rstrip() + " " + new_text.lstrip()).strip()
        return merged, {"mode": "fallback", "match_len": 0, "prev_keep": len(prev_toks), "new_start": 0}

    n, a0, b0 = best
    prev_keep = len(prev_toks) - len(prev_tail) + a0
    merged = prev_toks[:prev_keep] + new_toks[b0:]

    return _detokenize(merged), {"mode": "match", "match_len": n, "prev_keep": prev_keep, "new_start": b0}
