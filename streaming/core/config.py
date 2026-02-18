"""
Configuration management for the streaming pipeline.

Provides dataclasses for configuration and YAML loading utilities.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import yaml


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
MODEL_CACHE_DIR = PROJECT_ROOT / "model_cache"
REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 16000
    frame_ms: int = 100

    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)


@dataclass
class BufferConfig:
    """Buffer policy configuration."""
    type: str = "waitk"  # waitk, fixed, silence, vad
    k_s: float = 2.0
    hop_s: float = 1.0
    win_s: float = 6.0
    # For silence-terminated
    min_s: float = 0.5
    max_s: float = 10.0
    silence_s: float = 0.5
    silence_threshold: float = 0.01
    # For VAD buffer
    pre_s: float = 0.2
    post_s: float = 0.2
    min_speech_s: float = 0.4
    end_silence_s: float = 0.5


@dataclass
class ASRConfig:
    """ASR model configuration."""
    model_id: str = "openai/whisper-small"
    language: str = "en"
    task: str = "transcribe"
    device: Optional[str] = None
    max_new_tokens: int = 128


@dataclass
class ASRCommitConfig:
    """ASR commit policy configuration."""
    type: str = "stability"  # stability, immediate, windowed
    min_stable_steps: int = 2
    commit_words_behind: int = 5  # for windowed


@dataclass
class MTScopeConfig:
    """MT scope policy configuration."""
    type: str = "committed_plus_tail"  # committed_only, committed_plus_tail, full, delta
    tail_chars: int = 200


@dataclass
class MTConfig:
    """MT model configuration."""
    model_id: str = "Helsinki-NLP/opus-mt-en-de"
    src_lang: str = "en"
    tgt_lang: str = "de"
    device: Optional[str] = None
    max_length: int = 512
    num_beams: int = 4
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 0.1


@dataclass
class MTCommitConfig:
    """MT commit policy configuration."""
    type: str = "replace_pending"  # replace_pending, append_only, stability, full_replace
    min_stable_steps: int = 2  # for stability


@dataclass
class MetricsConfig:
    """Metrics computation configuration."""
    accuracy: List[str] = field(default_factory=lambda: ["wer"])
    quality: List[str] = field(default_factory=lambda: ["bleu", "comet_qe"])
    latency: List[str] = field(default_factory=lambda: ["al_words"])
    stability: List[str] = field(default_factory=lambda: ["normalized_erasure"])

    # Batch sizes
    comet_batch_size: int = 8
    comet_qe_batch_size: int = 8
    bertscore_batch_size: int = 16


@dataclass
class OutputConfig:
    """Output configuration."""
    console: bool = True
    jsonl: bool = True
    jsonl_path: Optional[Path] = None
    save_segments: bool = False
    segments_dir: Optional[Path] = None
    show_pending: bool = True
    show_timing: bool = True


@dataclass
class CommitConfig:
    """Combined commit configuration (for backward compatibility)."""
    asr_type: str = "stability"
    asr_min_stable_steps: int = 2
    mt_type: str = "replace_pending"
    mt_min_stable_steps: int = 2


@dataclass
class StreamingConfig:
    """
    Complete streaming pipeline configuration.

    Can be loaded from YAML or constructed programmatically.
    """
    audio: AudioConfig = field(default_factory=AudioConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    asr_commit: ASRCommitConfig = field(default_factory=ASRCommitConfig)
    mt_scope: MTScopeConfig = field(default_factory=MTScopeConfig)
    mt: MTConfig = field(default_factory=MTConfig)
    mt_commit: MTCommitConfig = field(default_factory=MTCommitConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Experiment tracking
    experiment_name: Optional[str] = None
    seed: int = 42
    verbose: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "StreamingConfig":
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamingConfig":
        """Create configuration from dictionary."""
        config = cls()

        if "audio" in data:
            config.audio = AudioConfig(**data["audio"])
        if "buffer" in data:
            config.buffer = BufferConfig(**data["buffer"])
        if "asr" in data:
            config.asr = ASRConfig(**data["asr"])
        if "asr_commit" in data:
            config.asr_commit = ASRCommitConfig(**data["asr_commit"])
        if "mt_scope" in data:
            config.mt_scope = MTScopeConfig(**data["mt_scope"])
        if "mt" in data:
            config.mt = MTConfig(**data["mt"])
        if "mt_commit" in data:
            config.mt_commit = MTCommitConfig(**data["mt_commit"])
        if "metrics" in data:
            config.metrics = MetricsConfig(**data["metrics"])
        if "output" in data:
            config.output = OutputConfig(**data["output"])

        if "experiment_name" in data:
            config.experiment_name = data["experiment_name"]
        if "seed" in data:
            config.seed = data["seed"]
        if "verbose" in data:
            config.verbose = data["verbose"]

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def save_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> StreamingConfig:
    """
    Load configuration with optional overrides.
    """
    if config_path is None:
        config_path = CONFIG_DIR / "default.yaml"

    config_path = Path(config_path)

    if config_path.exists():
        config = StreamingConfig.from_yaml(config_path)
    else:
        config = StreamingConfig()

    if overrides:
        config = _apply_overrides(config, overrides)

    return config


def _apply_overrides(config: StreamingConfig, overrides: Dict[str, Any]) -> StreamingConfig:
    """Apply nested overrides to configuration."""
    for key, value in overrides.items():
        if "." in key:
            parts = key.split(".")
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        elif hasattr(config, key):
            setattr(config, key, value)

    return config


def get_model_cache_path(subdir: str = "") -> Path:
    """Get path within model cache directory."""
    path = MODEL_CACHE_DIR / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path
