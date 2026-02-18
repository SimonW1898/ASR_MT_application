"""PySide6 client app for gRPC ASR->MT streaming demo."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import threading
import wave
from pathlib import Path
from typing import Iterator, Optional

try:
    import winsound
except Exception:  # pragma: no cover
    winsound = None

import grpc
import yaml

from .grpc_client import DEFAULT_HOST, DEFAULT_PORT, _build_config, _load_config
from .ffmpeg_mic import FFmpegMicSource
from .wav_streamer import WavFileSource

try:
    from grpc_demo.proto_gen import asrmt_pb2, asrmt_pb2_grpc
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing generated proto code. Generate grpc_demo/proto_gen/*.py first."
    ) from exc

try:
    from PySide6.QtCore import QThread, Signal, QTimer, Qt
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QCheckBox,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "PySide6 is required for the GUI client. Install with: pip install PySide6"
    ) from exc


DEFAULT_MODEL_CACHE_PLAN = "grpc_demo/configs/model_cache_plan.json"
DEFAULT_STREAM_DIR = "data/stream"
LANGUAGE_ALIASES = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "de": "de",
    "ger": "de",
    "german": "de",
    "fr": "fr",
    "fra": "fr",
    "french": "fr",
    "ar": "ar",
    "ara": "ar",
    "arabic": "ar",
    "ja": "ja",
    "jpn": "ja",
    "japanese": "ja",
    "fa": "fa",
    "fas": "fa",
    "farsi": "fa",
    "persian": "fa",
}


def _pair_label(source: str, target: str) -> str:
    return f"OPUS {source.upper()}-{target.upper()}"


def _load_model_cache_plan(config_path: str) -> tuple[
    dict[str, list[tuple[str, str]]],
    list[tuple[str, str]],
    dict[tuple[str, str], list[tuple[str, str]]],
]:
    plan_path = Path(config_path).resolve().parent / "model_cache_plan.json"
    if not plan_path.exists():
        plan_path = Path(DEFAULT_MODEL_CACHE_PLAN)

    try:
        with open(plan_path, "r", encoding="utf-8") as f:
            plan = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return ({"en": [("Whisper Small", "openai/whisper-small")]}, [("Whisper Small", "openai/whisper-small")], {("en", "de"): [("OPUS EN-DE", "Helsinki-NLP/opus-mt-en-de")]})

    asr_models_by_lang: dict[str, list[tuple[str, str]]] = {}
    asr_models_all: list[tuple[str, str]] = []
    mt_models_by_pair: dict[tuple[str, str], list[tuple[str, str]]] = {}

    for item in plan.get("asr_models", []):
        model_id = str(item.get("model_id", "")).strip()
        if not model_id:
            continue
        name = str(item.get("name", model_id)).strip() or model_id
        asr_lang = str(item.get("asr_language", "")).strip().lower()
        model_tuple = (name, model_id)
        if model_tuple not in asr_models_all:
            asr_models_all.append(model_tuple)
        if asr_lang:
            asr_models_by_lang.setdefault(asr_lang, [])
            if model_tuple not in asr_models_by_lang[asr_lang]:
                asr_models_by_lang[asr_lang].append(model_tuple)

    for item in plan.get("mt_models", []):
        model_id = str(item.get("model_id", "")).strip()
        pair = str(item.get("pair", "")).strip().lower()
        if not model_id or "-" not in pair:
            continue
        source, target = pair.split("-", 1)
        source = source.strip()
        target = target.strip()
        if not source or not target:
            continue
        label = str(item.get("name", _pair_label(source, target))).strip() or _pair_label(source, target)
        mt_models_by_pair.setdefault((source, target), [])
        model_tuple = (label, model_id)
        if model_tuple not in mt_models_by_pair[(source, target)]:
            mt_models_by_pair[(source, target)].append(model_tuple)

    if not asr_models_all:
        asr_models_all = [("Whisper Small", "openai/whisper-small")]
    if not mt_models_by_pair:
        mt_models_by_pair = {("en", "de"): [("OPUS EN-DE", "Helsinki-NLP/opus-mt-en-de")]}

    return asr_models_by_lang, asr_models_all, mt_models_by_pair


class StreamingWorker(QThread):
    """Background gRPC streaming worker for GUI responsiveness."""

    status_signal = Signal(str, str)
    segment_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(
        self,
        config_path: str,
        host_override: Optional[str] = None,
        port_override: Optional[int] = None,
        use_microphone_override: Optional[bool] = None,
        device_hint_override: Optional[str] = None,
        asr_language_override: Optional[str] = None,
        asr_model_override: Optional[str] = None,
        mt_model_override: Optional[str] = None,
    ):
        super().__init__()
        self.config_path = config_path
        self.host_override = host_override
        self.port_override = port_override
        self.use_microphone_override = use_microphone_override
        self.device_hint_override = device_hint_override
        self.asr_language_override = asr_language_override
        self.asr_model_override = asr_model_override
        self.mt_model_override = mt_model_override
        self._stop_requested = False
        self._channel: Optional[grpc.Channel] = None
        self._ready_to_stream = threading.Event()

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            self._ready_to_stream.clear()
            cfg = self._safe_load_config(self.config_path)
            cfg_msg = _build_config(cfg)

            input_mode = dict(cfg.get("input_mode", {}))
            if self.use_microphone_override is not None:
                input_mode["use_microphone"] = self.use_microphone_override
            if self.device_hint_override is not None:
                input_mode["device_hint"] = self.device_hint_override
            runtime = cfg.get("runtime", {})

            cfg_msg.input_mode.use_microphone = bool(input_mode.get("use_microphone", False))
            cfg_msg.input_mode.device_hint = str(input_mode.get("device_hint", ""))

            if self.asr_language_override:
                cfg_msg.models.asr_language = self.asr_language_override
            if self.asr_model_override:
                cfg_msg.models.asr_model_id = self.asr_model_override
            if self.mt_model_override:
                cfg_msg.models.mt_model_id = self.mt_model_override

            cfg_msg.runtime.offline = True

            host = self.host_override or str(runtime.get("grpc_host", DEFAULT_HOST))
            port = self.port_override or int(runtime.get("grpc_port", DEFAULT_PORT))

            self._channel = grpc.insecure_channel(f"{host}:{port}")
            stub = asrmt_pb2_grpc.ASRMTServiceStub(self._channel)

            responses = stub.StreamSession(
                self._request_iter(
                    cfg_msg,
                    input_mode=input_mode,
                )
            )

            for response in responses:
                if response.HasField("status"):
                    status = response.status
                    self.status_signal.emit(status.state, status.message)
                    if str(status.state).upper() == "READY":
                        self._ready_to_stream.set()
                if response.HasField("segment"):
                    segment = response.segment
                    self.segment_signal.emit(
                        {
                            "segment_id": int(segment.segment_id),
                            "asr_chunk_text": segment.asr_chunk_text,
                            "mt_chunk_text": segment.mt_chunk_text,
                            "merged_asr": segment.merged_asr,
                            "merged_mt": segment.merged_mt,
                            "reason": segment.reason,
                            "audio_time_s": float(segment.audio_time_s),
                            "queue_latency_ms": float(segment.queue_latency_ms),
                            "process_time_ms": float(segment.process_time_ms),
                            "e2e_ms": float(segment.e2e_ms),
                        }
                    )

                if self._stop_requested:
                    break
        except Exception as exc:
            self.error_signal.emit(str(exc))
        finally:
            if self._channel is not None:
                self._channel.close()
                self._channel = None

    def _request_iter(
        self,
        cfg_msg: asrmt_pb2.SessionConfig,
        input_mode: dict,
    ) -> Iterator[asrmt_pb2.ClientMessage]:
        yield asrmt_pb2.ClientMessage(config=cfg_msg)

        wait_deadline = time.time() + 60.0
        while (
            not self._ready_to_stream.is_set()
            and not self._stop_requested
            and time.time() < wait_deadline
        ):
            time.sleep(0.05)

        if not self._ready_to_stream.is_set() and not self._stop_requested:
            self.status_signal.emit("WARN", "READY timeout; starting stream anyway")

        use_microphone = bool(input_mode.get("use_microphone", False))
        if use_microphone:
            source = FFmpegMicSource(
                sample_rate=cfg_msg.audio.sample_rate,
                chunk_ms=cfg_msg.audio.transport_chunk_ms,
                device_hint=str(input_mode.get("device_hint", "")),
            )
            for frame in source.frames(stop_requested=lambda: self._stop_requested):
                if self._stop_requested:
                    break
                yield asrmt_pb2.ClientMessage(
                    audio=asrmt_pb2.AudioChunk(
                        pcm16=frame.pcm16,
                        t0=frame.t0,
                        t1=frame.t1,
                        frame_idx=frame.frame_idx,
                    )
                )
            yield asrmt_pb2.ClientMessage(control=asrmt_pb2.Control(action="STOP"))
            return

        wav_path = str(input_mode.get("wav_path", "data/stream/demo.wav"))
        realtime_simulation = bool(input_mode.get("realtime_simulation", True))
        source = WavFileSource(
            wav_path=wav_path,
            sample_rate=cfg_msg.audio.sample_rate,
            chunk_ms=cfg_msg.audio.transport_chunk_ms,
            realtime_simulation=realtime_simulation,
        )

        for frame in source.frames():
            if self._stop_requested:
                break
            yield asrmt_pb2.ClientMessage(
                audio=asrmt_pb2.AudioChunk(
                    pcm16=frame.pcm16,
                    t0=frame.t0,
                    t1=frame.t1,
                    frame_idx=frame.frame_idx,
                )
            )

        yield asrmt_pb2.ClientMessage(control=asrmt_pb2.Control(action="STOP"))

    @staticmethod
    def _safe_load_config(path: str) -> dict:
        try:
            return _load_config(path)
        except (FileNotFoundError, yaml.YAMLError):
            return {}


class MainWindow(QMainWindow):
    """PySide6 main window for the streaming client."""

    def __init__(self, default_config: str):
        super().__init__()
        self.setWindowTitle("ASR->MT gRPC Demo (PySide6)")
        self.resize(1100, 700)

        self.worker: Optional[StreamingWorker] = None
        self.recent_e2e_ms: list[float] = []
        self.reference_window: Optional[QMainWindow] = None
        self.reference_view: Optional[QTextEdit] = None
        self._playback_thread: Optional[threading.Thread] = None
        self._playback_stop_event = threading.Event()
        self._pending_wav_path: Optional[Path] = None
        self._pending_target_lang: str = "de"
        self._ready_actions_done = False

        self.session_elapsed_seconds = 0
        self.session_timer = QTimer(self)
        self.session_timer.setInterval(1000)
        self.session_timer.timeout.connect(self._on_session_timer_tick)

        self._apply_visual_style()

        root = QWidget(self)
        layout = QVBoxLayout(root)

        controls_box = QGroupBox("Session Setup")
        controls_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        controls_box.setMaximumHeight(250)
        controls = QGridLayout(controls_box)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(6)
        controls.setColumnStretch(2, 3)
        controls.setColumnStretch(4, 3)
        controls.setColumnStretch(6, 1)

        self.config_edit = QLineEdit(default_config)
        self.browse_button = QPushButton("Browse")
        self.browse_button.setObjectName("secondaryButton")
        self.host_edit = QLineEdit("")
        self.host_edit.setPlaceholderText("Optional override (default from YAML)")
        self.port_edit = QLineEdit("")
        self.port_edit.setPlaceholderText("Optional override (default from YAML)")
        self.use_microphone_checkbox = QCheckBox("")
        self.use_microphone_checkbox.setToolTip("Checked = live microphone input; unchecked = stream selected WAV file")
        self.device_hint_edit = QLineEdit("")
        self.device_hint_edit.setPlaceholderText("Device hint (optional, substring match)")
        self.device_hint_edit.setToolTip("Only used in microphone mode")
        self.list_devices_button = QPushButton("List devices")
        self.list_devices_button.setObjectName("secondaryButton")
        self.wav_file_combo = QComboBox()
        self.wav_file_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.wav_file_combo.setToolTip("Select WAV input file for live-simulated streaming")
        self.browse_wav_button = QPushButton("Browse WAV")
        self.browse_wav_button.setObjectName("secondaryButton")
        self.wav_duration_label = QLabel("Max: -")
        self.wav_duration_label.setMinimumWidth(90)

        self.source_lang_combo = QComboBox()
        self.target_lang_combo = QComboBox()
        self.asr_model_combo = QComboBox()
        self.mt_model_combo = QComboBox()

        (
            self._asr_models_by_lang,
            self._asr_models_all,
            self._mt_models_by_pair,
        ) = _load_model_cache_plan(default_config)

        self._source_langs: list[str] = sorted({src for (src, _tgt) in self._mt_models_by_pair.keys()})
        if not self._source_langs:
            self._source_langs = ["en"]
        for lang in self._source_langs:
            self.source_lang_combo.addItem(lang)

        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("startButton")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setEnabled(False)

        left_labels = {
            0: "Config YAML",
            1: "Connection",
            2: "Input mode",
            3: "Input file",
            4: "Language",
            5: "Model",
            6: "Actions",
        }
        for row, text in left_labels.items():
            label = QLabel(text)
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            controls.addWidget(label, row, 0)

        controls.addWidget(self.config_edit, 0, 2, 1, 3)
        controls.addWidget(self.browse_button, 0, 5)

        controls.addWidget(QLabel("Host"), 1, 1)
        controls.addWidget(self.host_edit, 1, 2)
        controls.addWidget(QLabel("Port"), 1, 3)
        controls.addWidget(self.port_edit, 1, 4)

        controls.addWidget(QLabel("Microphone"), 2, 1)
        controls.addWidget(self.use_microphone_checkbox, 2, 2)
        controls.addWidget(QLabel("Mic device"), 2, 3)
        controls.addWidget(self.device_hint_edit, 2, 4)
        controls.addWidget(self.list_devices_button, 2, 5)

        controls.addWidget(self.wav_file_combo, 3, 2, 1, 3)
        controls.addWidget(self.browse_wav_button, 3, 5)
        controls.addWidget(self.wav_duration_label, 3, 6)

        controls.addWidget(QLabel("Source"), 4, 1)
        controls.addWidget(self.source_lang_combo, 4, 2)
        controls.addWidget(QLabel("Target"), 4, 3)
        controls.addWidget(self.target_lang_combo, 4, 4)

        controls.addWidget(QLabel("ASR"), 5, 1)
        controls.addWidget(self.asr_model_combo, 5, 2)
        controls.addWidget(QLabel("MT"), 5, 3)
        controls.addWidget(self.mt_model_combo, 5, 4)

        self.source_lang_combo.setToolTip("Language for speech recognition")
        self.target_lang_combo.setToolTip("Translation target language")
        self.asr_model_combo.setToolTip("ASR model options filtered by source language")
        self.mt_model_combo.setToolTip("MT model options filtered by source-target pair")

        self.config_edit.setMinimumWidth(380)
        self.host_edit.setMinimumWidth(260)
        self.device_hint_edit.setMinimumWidth(260)
        self.port_edit.setMinimumWidth(100)
        self.port_edit.setMaximumWidth(120)
        self.source_lang_combo.setMinimumWidth(220)
        self.target_lang_combo.setMinimumWidth(220)
        self.asr_model_combo.setMinimumWidth(220)
        self.mt_model_combo.setMinimumWidth(220)
        self.browse_button.setFixedWidth(100)
        self.browse_wav_button.setFixedWidth(100)
        self.list_devices_button.setFixedWidth(100)
        self.start_button.setFixedWidth(96)
        self.stop_button.setFixedWidth(96)

        controls.addWidget(self.start_button, 6, 5)
        controls.addWidget(self.stop_button, 6, 6)

        metrics_box = QGroupBox("Status & Metrics")
        metrics_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        metrics_box.setMaximumHeight(90)
        metrics_layout = QHBoxLayout(metrics_box)
        self.status_label = QLabel("Idle")
        self.timer_label = QLabel("Elapsed: 00:00")
        self.last_metrics_label = QLabel("Last: -")
        self.avg_metrics_label = QLabel("Rolling avg (20): -")
        metrics_layout.addWidget(self.status_label, 2)
        metrics_layout.addWidget(self.timer_label, 1)
        metrics_layout.addWidget(self.last_metrics_label, 2)
        metrics_layout.addWidget(self.avg_metrics_label, 2)

        outputs_box = QGroupBox("Transcripts")
        outputs_layout = QGridLayout(outputs_box)
        self.asr_view = QTextEdit()
        self.asr_view.setReadOnly(True)
        self.asr_view.setMinimumHeight(300)
        self.mt_view = QTextEdit()
        self.mt_view.setReadOnly(True)
        self.mt_view.setMinimumHeight(300)
        outputs_layout.addWidget(QLabel("Merged ASR"), 0, 0)
        outputs_layout.addWidget(QLabel("Merged MT"), 0, 1)
        outputs_layout.addWidget(self.asr_view, 1, 0)
        outputs_layout.addWidget(self.mt_view, 1, 1)

        events_box = QGroupBox("Events")
        events_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        events_layout = QVBoxLayout(events_box)
        self.events_view = QTextEdit()
        self.events_view.setReadOnly(True)
        self.events_view.setMinimumHeight(120)
        events_layout.addWidget(self.events_view)

        layout.addWidget(controls_box, 0)
        layout.addWidget(metrics_box, 0)
        layout.addWidget(outputs_box, 8)
        layout.addWidget(events_box, 1)

        self.setCentralWidget(root)

        self.browse_button.clicked.connect(self._on_browse)
        self.browse_wav_button.clicked.connect(self._on_browse_wav)
        self.start_button.clicked.connect(self._on_start)
        self.stop_button.clicked.connect(self._on_stop)
        self.list_devices_button.clicked.connect(self._on_list_devices)
        self.source_lang_combo.currentIndexChanged.connect(self._refresh_model_combos)
        self.target_lang_combo.currentIndexChanged.connect(self._refresh_model_combos)
        self.use_microphone_checkbox.stateChanged.connect(self._update_input_mode_ui)
        self.wav_file_combo.currentIndexChanged.connect(self._on_wav_selection_changed)
        self._sync_input_controls_from_config(default_config)

    def _apply_visual_style(self) -> None:
        self.setStyleSheet(
            """
            QGroupBox {
                font-weight: 600;
                margin-top: 10px;
                border: 1px solid #3b3f46;
                border-radius: 8px;
                padding: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QComboBox, QLineEdit, QPushButton {
                min-height: 24px;
                padding: 2px 6px;
                border-radius: 6px;
            }
            QComboBox, QLineEdit {
                border: 1px solid #565d68;
            }
            QPushButton#startButton {
                background-color: #1f8b4c;
                color: white;
                font-weight: 700;
                border: 1px solid #1a7340;
            }
            QPushButton#startButton:hover {
                background-color: #25a35a;
            }
            QPushButton#startButton:disabled {
                background-color: #6f987f;
            }
            QPushButton#stopButton {
                background-color: #bf2f45;
                color: white;
                font-weight: 700;
                border: 1px solid #9b2437;
            }
            QPushButton#stopButton:hover {
                background-color: #d73650;
            }
            QPushButton#stopButton:disabled {
                background-color: #9f7a80;
            }
            QPushButton#secondaryButton {
                background-color: #2f6fed;
                color: white;
                font-weight: 700;
                border: 1px solid #285fca;
                min-width: 96px;
            }
            QPushButton#secondaryButton:hover {
                background-color: #3e7cf3;
            }
            QPushButton#secondaryButton:disabled {
                background-color: #6f83aa;
            }
            QTextEdit {
                border-radius: 6px;
            }
            """
        )

    @staticmethod
    def _default_stream_dir() -> Path:
        return Path.cwd() / DEFAULT_STREAM_DIR

    @staticmethod
    def _to_repo_relative(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(Path.cwd().resolve())).replace("\\", "/")
        except ValueError:
            return str(path)

    def _refresh_wav_files(self, selected_wav: Optional[str] = None) -> None:
        stream_dir = self._default_stream_dir()
        stream_dir.mkdir(parents=True, exist_ok=True)
        wav_files = sorted(stream_dir.glob("*.wav"))

        selected_abs = ""
        if selected_wav:
            selected_abs = str((Path(selected_wav) if Path(selected_wav).is_absolute() else Path.cwd() / selected_wav).resolve())

        self.wav_file_combo.blockSignals(True)
        self.wav_file_combo.clear()

        for wav_path in wav_files:
            self.wav_file_combo.addItem(wav_path.name, str(wav_path.resolve()))

        if self.wav_file_combo.count() == 0:
            self.wav_file_combo.addItem("(no WAV files found in data/stream)", "")
            self.wav_file_combo.setEnabled(False)
        else:
            self.wav_file_combo.setEnabled(True)
            if selected_abs:
                idx = self.wav_file_combo.findData(selected_abs)
                if idx >= 0:
                    self.wav_file_combo.setCurrentIndex(idx)

        self.wav_file_combo.blockSignals(False)
        self._on_wav_selection_changed()

    def _on_wav_selection_changed(self) -> None:
        self._update_wav_duration_label()
        self._apply_wav_based_model_selection()

    @staticmethod
    def _normalize_lang_token(token: str) -> Optional[str]:
        key = token.strip().lower()
        return LANGUAGE_ALIASES.get(key)

    def _infer_lang_pair_from_wav(self, wav_path: Path) -> Optional[tuple[str, str]]:
        parts = [wav_path.stem, wav_path.parent.name]
        candidates: list[str] = []
        for part in parts:
            for token in re.split(r"[^a-zA-Z0-9]+", part.lower()):
                if token:
                    candidates.append(token)

        langs: list[str] = []
        for token in candidates:
            normalized = self._normalize_lang_token(token)
            if normalized:
                langs.append(normalized)

        if len(langs) < 2:
            return None
        return langs[-2], langs[-1]

    def _apply_wav_based_model_selection(self) -> None:
        wav_path = self._selected_wav_abs_path()
        if wav_path is None:
            return

        inferred = self._infer_lang_pair_from_wav(wav_path)
        if inferred is None:
            return

        source_lang, target_lang = inferred
        src_idx = self.source_lang_combo.findText(source_lang)
        if src_idx < 0:
            return

        self.source_lang_combo.blockSignals(True)
        self.source_lang_combo.setCurrentIndex(src_idx)
        self.source_lang_combo.blockSignals(False)

        self._refresh_model_combos()

        tgt_idx = self.target_lang_combo.findText(target_lang)
        if tgt_idx >= 0:
            self.target_lang_combo.blockSignals(True)
            self.target_lang_combo.setCurrentIndex(tgt_idx)
            self.target_lang_combo.blockSignals(False)
            self._refresh_model_combos()

        if self.asr_model_combo.count() > 0:
            self.asr_model_combo.setCurrentIndex(0)
        if self.mt_model_combo.count() > 0:
            self.mt_model_combo.setCurrentIndex(0)

    def _update_wav_duration_label(self) -> None:
        wav_path = self._selected_wav_abs_path()
        if wav_path is None:
            self.wav_duration_label.setText("Max: -")
            return
        try:
            with wave.open(str(wav_path), "rb") as wav_file:
                frame_rate = wav_file.getframerate()
                frame_count = wav_file.getnframes()
            if frame_rate <= 0:
                raise ValueError("Invalid frame rate")
            seconds = frame_count / frame_rate
            self.wav_duration_label.setText(f"Max: {seconds:.1f}s")
        except Exception:
            self.wav_duration_label.setText("Max: unknown")

    def _selected_wav_abs_path(self) -> Optional[Path]:
        wav_data = self.wav_file_combo.currentData()
        if not wav_data:
            return None
        wav_path = Path(str(wav_data))
        if not wav_path.exists():
            return None
        return wav_path

    def _reference_json_for_wav(self, wav_path: Path) -> Path:
        return wav_path.with_suffix(".json")

    def _load_reference_text(self, wav_path: Path, target_lang: str) -> str:
        json_path = self._reference_json_for_wav(wav_path)
        if not json_path.exists():
            return f"No sidecar reference JSON found for:\n{wav_path.name}\n\nExpected:\n{json_path.name}"

        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            return f"Failed to load reference JSON:\n{json_path.name}\n\nError: {exc}"

        segments = payload.get("segments", [])
        lines: list[str] = []
        target_key = f"tgt_ref_{target_lang.lower()}"

        for item in segments:
            if str(item.get("kind", "")).lower() != "utt":
                continue
            text = str(item.get(target_key) or item.get("tgt_ref") or "").strip()
            if not text:
                continue
            lines.append(text)

        if not lines:
            return (
                f"No target references found in:\n{json_path.name}\n\n"
                f"Expected one of: '{target_key}' or 'tgt_ref' on utterance segments."
            )

        return "\n".join(f"{idx + 1}. {line}" for idx, line in enumerate(lines))

    def _ensure_reference_window(self) -> None:
        if self.reference_window is not None:
            return
        window = QMainWindow(self)
        window.setWindowTitle("Reference Translation")
        window.resize(700, 600)
        view = QTextEdit(window)
        view.setReadOnly(True)
        window.setCentralWidget(view)
        self.reference_window = window
        self.reference_view = view

    def _show_reference_window(self, wav_path: Path, target_lang: str) -> None:
        self._ensure_reference_window()
        assert self.reference_window is not None
        assert self.reference_view is not None
        self.reference_window.setWindowTitle(
            f"Reference Translation ({target_lang.lower()}) - {wav_path.name}"
        )
        self.reference_view.setPlainText(self._load_reference_text(wav_path, target_lang))
        self.reference_window.show()
        self.reference_window.raise_()

    def _hide_reference_window(self) -> None:
        if self.reference_window is not None:
            self.reference_window.hide()

    def _start_wav_playback(self, wav_path: Path) -> None:
        self._stop_wav_playback()
        if winsound is None:
            self._append_event("[AUDIO] winsound unavailable; WAV playback skipped")
            return

        self._playback_stop_event.clear()

        def _play() -> None:
            try:
                winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)
            except Exception as exc:
                self._append_event(f"[AUDIO] Playback failed: {exc}")

        self._playback_thread = threading.Thread(target=_play, daemon=True)
        self._playback_thread.start()

    def _stop_wav_playback(self) -> None:
        self._playback_stop_event.set()
        if winsound is not None:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

    def _on_browse(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select config YAML",
            str(Path.cwd()),
            "YAML files (*.yaml *.yml)",
        )
        if selected:
            self.config_edit.setText(selected)
            (
                self._asr_models_by_lang,
                self._asr_models_all,
                self._mt_models_by_pair,
            ) = _load_model_cache_plan(selected)
            self._source_langs = sorted({src for (src, _tgt) in self._mt_models_by_pair.keys()})
            if not self._source_langs:
                self._source_langs = ["en"]
            self.source_lang_combo.blockSignals(True)
            self.source_lang_combo.clear()
            for lang in self._source_langs:
                self.source_lang_combo.addItem(lang)
            self.source_lang_combo.blockSignals(False)
            self._sync_input_controls_from_config(selected)

    def _on_browse_wav(self) -> None:
        start_dir = self._default_stream_dir()
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select WAV file",
            str(start_dir),
            "WAV files (*.wav)",
        )
        if not selected:
            return

        selected_path = Path(selected).resolve()
        self._refresh_wav_files(str(selected_path))

        if self.wav_file_combo.findData(str(selected_path)) < 0:
            self.wav_file_combo.addItem(selected_path.name, str(selected_path))
            self.wav_file_combo.setCurrentIndex(self.wav_file_combo.count() - 1)

    def _on_start(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return

        self._clear_session_outputs()

        config_path = self.config_edit.text().strip() or "grpc_demo/configs/demo.yaml"
        host_override = self.host_edit.text().strip() or None
        port_text = self.port_edit.text().strip()

        port_override: Optional[int] = None
        if port_text:
            try:
                port_override = int(port_text)
            except ValueError:
                QMessageBox.critical(self, "Invalid Port", "Port must be an integer.")
                return

        self._persist_gui_to_config(
            config_path=config_path,
            host_override=host_override,
            port_override=port_override,
        )

        self._pending_wav_path = None
        self._pending_target_lang = self.target_lang_combo.currentText().strip() or "de"
        self._ready_actions_done = False

        if not self.use_microphone_checkbox.isChecked():
            wav_path = self._selected_wav_abs_path()
            if wav_path is None:
                QMessageBox.critical(
                    self,
                    "Missing WAV File",
                    "Select a valid WAV file for file-based streaming.",
                )
                return
            self._pending_wav_path = wav_path
            self._append_event(f"[AUDIO] WAV queued: {wav_path.name} (starts on READY)")
        else:
            self._stop_wav_playback()
            self._hide_reference_window()

        self.recent_e2e_ms.clear()

        if self.asr_model_combo.count() == 0 or self.mt_model_combo.count() == 0:
            QMessageBox.critical(
                self,
                "Invalid Model Selection",
                "No valid ASR/MT model combination is available for the selected language pair from model_cache_plan.json.",
            )
            return

        self.worker = StreamingWorker(
            config_path=config_path,
            host_override=host_override,
            port_override=port_override,
            use_microphone_override=self.use_microphone_checkbox.isChecked(),
            device_hint_override=self.device_hint_edit.text().strip(),
            asr_language_override=self.source_lang_combo.currentText().strip() or None,
            asr_model_override=self.asr_model_combo.currentData(),
            mt_model_override=self.mt_model_combo.currentData(),
        )
        self.worker.status_signal.connect(self._on_status)
        self.worker.segment_signal.connect(self._on_segment)
        self.worker.error_signal.connect(self._on_error)
        self.worker.finished.connect(self._on_worker_finished)

        self._set_running(True)
        self._reset_session_timer()
        self.status_label.setText("CONNECTING: waiting for server READY")
        self._append_event(f"[START] config={config_path}")
        self.worker.start()

    def _clear_session_outputs(self) -> None:
        self.asr_view.clear()
        self.mt_view.clear()
        self.events_view.clear()
        self.recent_e2e_ms.clear()
        self.last_metrics_label.setText("Last: -")
        self.avg_metrics_label.setText("Rolling avg (20): -")
        self._reset_session_timer()

    def _persist_gui_to_config(
        self,
        config_path: str,
        host_override: Optional[str],
        port_override: Optional[int],
    ) -> None:
        cfg = StreamingWorker._safe_load_config(config_path)
        if not cfg:
            cfg = {}

        input_mode = cfg.setdefault("input_mode", {})
        input_mode["use_microphone"] = self.use_microphone_checkbox.isChecked()
        input_mode["device_hint"] = self.device_hint_edit.text().strip()
        selected_wav = self._selected_wav_abs_path()
        if selected_wav is not None:
            input_mode["wav_path"] = self._to_repo_relative(selected_wav)

        models = cfg.setdefault("models", {})
        models["asr_language"] = self.source_lang_combo.currentText().strip() or "en"
        asr_model = self.asr_model_combo.currentData()
        if asr_model:
            models["asr_model_id"] = str(asr_model)
        mt_model = self.mt_model_combo.currentData()
        if mt_model:
            models["mt_model_id"] = str(mt_model)

        runtime = cfg.setdefault("runtime", {})
        runtime["offline"] = True
        if host_override:
            runtime["grpc_host"] = host_override
        if port_override is not None:
            runtime["grpc_port"] = port_override

        ui = cfg.setdefault("ui", {})
        ui["target_language"] = self.target_lang_combo.currentText().strip() or "de"

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        self._append_event("[CONFIG] Saved GUI selections to YAML")

    def _on_stop(self) -> None:
        if self.worker is None:
            return
        self._append_event("[CONTROL] STOP requested")
        self._stop_wav_playback()
        self._stop_session_timer()
        self.worker.request_stop()

    def _on_status(self, state: str, message: str) -> None:
        self.status_label.setText(f"{state}: {message}")
        self._append_event(f"[STATUS] {state}: {message}")
        if str(state).upper() == "READY":
            self._on_stream_ready()

    def _on_stream_ready(self) -> None:
        if self._ready_actions_done:
            return
        self._ready_actions_done = True
        self._start_session_timer()
        if self._pending_wav_path is not None:
            self._show_reference_window(
                wav_path=self._pending_wav_path,
                target_lang=self._pending_target_lang,
            )
            self._start_wav_playback(self._pending_wav_path)
            self._append_event(f"[AUDIO] Playing WAV: {self._pending_wav_path.name}")

    def _reset_session_timer(self) -> None:
        self.session_elapsed_seconds = 0
        self.timer_label.setText("Elapsed: 00:00")

    def _start_session_timer(self) -> None:
        if not self.session_timer.isActive():
            self.session_timer.start()

    def _stop_session_timer(self) -> None:
        if self.session_timer.isActive():
            self.session_timer.stop()

    def _on_session_timer_tick(self) -> None:
        self.session_elapsed_seconds += 1
        minutes = self.session_elapsed_seconds // 60
        seconds = self.session_elapsed_seconds % 60
        self.timer_label.setText(f"Elapsed: {minutes:02d}:{seconds:02d}")

    def _on_segment(self, seg: dict) -> None:
        self.asr_view.setPlainText(seg["merged_asr"])
        self.mt_view.setPlainText(seg["merged_mt"])

        e2e = seg.get("e2e_ms", 0.0)
        self.recent_e2e_ms.append(e2e)
        self.recent_e2e_ms = self.recent_e2e_ms[-20:]

        self.last_metrics_label.setText(
            "Last: "
            f"queue={seg['queue_latency_ms']:.1f}ms "
            f"process={seg['process_time_ms']:.1f}ms "
            f"e2e={e2e:.1f}ms"
        )
        avg_e2e = statistics.fmean(self.recent_e2e_ms) if self.recent_e2e_ms else 0.0
        self.avg_metrics_label.setText(f"Rolling avg (20): e2e={avg_e2e:.1f}ms")

        self._append_event(
            f"[SEGMENT] id={seg['segment_id']} reason={seg['reason']} audio_t={seg['audio_time_s']:.2f}s"
        )

    def _on_error(self, message: str) -> None:
        self._append_event(f"[ERROR] {message}")
        QMessageBox.critical(self, "Streaming Error", message)

    def _on_worker_finished(self) -> None:
        self._set_running(False)
        self._stop_wav_playback()
        self._stop_session_timer()
        self._pending_wav_path = None
        self._ready_actions_done = False
        self._append_event("[STOP] Worker finished")

    def _append_event(self, text: str) -> None:
        self.events_view.append(text)

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.browse_button.setEnabled(not running)
        self.config_edit.setEnabled(not running)
        self.host_edit.setEnabled(not running)
        self.port_edit.setEnabled(not running)
        self.use_microphone_checkbox.setEnabled(not running)
        self.device_hint_edit.setEnabled(not running)
        self.list_devices_button.setEnabled(not running)
        self.wav_file_combo.setEnabled(not running and not self.use_microphone_checkbox.isChecked())
        self.browse_wav_button.setEnabled(not running and not self.use_microphone_checkbox.isChecked())
        self.source_lang_combo.setEnabled(not running)
        self.target_lang_combo.setEnabled(not running)
        self.asr_model_combo.setEnabled(not running)
        self.mt_model_combo.setEnabled(not running)

    def _update_input_mode_ui(self) -> None:
        using_mic = self.use_microphone_checkbox.isChecked()
        running = self.worker is not None and self.worker.isRunning()
        self.device_hint_edit.setEnabled(not running and using_mic)
        self.list_devices_button.setEnabled(not running and using_mic)
        self.wav_file_combo.setEnabled(not running and not using_mic and self.wav_file_combo.count() > 0)
        self.browse_wav_button.setEnabled(not running and not using_mic)

        if using_mic:
            self._hide_reference_window()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._stop_wav_playback()
        self._stop_session_timer()
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait(2000)
        super().closeEvent(event)

    def _sync_input_controls_from_config(self, path: str) -> None:
        try:
            cfg = _load_config(path)
        except (FileNotFoundError, yaml.YAMLError):
            self._refresh_model_combos()
            return
        input_mode = cfg.get("input_mode", {})
        model_cfg = cfg.get("models", {})

        self.use_microphone_checkbox.setChecked(bool(input_mode.get("use_microphone", False)))
        self.device_hint_edit.setText(str(input_mode.get("device_hint", "")))
        self._refresh_wav_files(str(input_mode.get("wav_path", "")))

        asr_lang = str(model_cfg.get("asr_language", "en"))
        src_idx = self.source_lang_combo.findText(asr_lang)
        if src_idx >= 0:
            self.source_lang_combo.setCurrentIndex(src_idx)

        ui_cfg = cfg.get("ui", {})
        target_lang = str(ui_cfg.get("target_language", "")).strip().lower()

        self._refresh_model_combos()

        if target_lang:
            tgt_idx = self.target_lang_combo.findText(target_lang)
            if tgt_idx >= 0:
                self.target_lang_combo.setCurrentIndex(tgt_idx)
                self._refresh_model_combos()

        asr_model_id = str(model_cfg.get("asr_model_id", ""))
        mt_model_id = str(model_cfg.get("mt_model_id", ""))
        asr_idx = self.asr_model_combo.findData(asr_model_id)
        if asr_idx >= 0:
            self.asr_model_combo.setCurrentIndex(asr_idx)
        mt_idx = self.mt_model_combo.findData(mt_model_id)
        if mt_idx >= 0:
            self.mt_model_combo.setCurrentIndex(mt_idx)

        self._update_input_mode_ui()

    def _refresh_model_combos(self) -> None:
        source_lang = self.source_lang_combo.currentText().strip() or "en"
        prev_target = self.target_lang_combo.currentText().strip()

        prev_asr = self.asr_model_combo.currentData()
        prev_mt = self.mt_model_combo.currentData()

        available_targets = sorted(
            {tgt for (src, tgt) in self._mt_models_by_pair.keys() if src == source_lang}
        )
        if not available_targets:
            available_targets = ["de"]

        self.target_lang_combo.blockSignals(True)
        self.target_lang_combo.clear()
        for lang in available_targets:
            self.target_lang_combo.addItem(lang)
        target_keep = prev_target if prev_target in available_targets else available_targets[0]
        target_idx = self.target_lang_combo.findText(target_keep)
        if target_idx >= 0:
            self.target_lang_combo.setCurrentIndex(target_idx)
        self.target_lang_combo.blockSignals(False)

        target_lang = self.target_lang_combo.currentText().strip() or available_targets[0]

        self.asr_model_combo.clear()
        asr_options = self._asr_models_by_lang.get(source_lang, self._asr_models_all)
        for label, model_id in asr_options:
            self.asr_model_combo.addItem(label, model_id)
        if self.asr_model_combo.count() == 0:
            self.asr_model_combo.addItem("Whisper Small", "openai/whisper-small")

        self.mt_model_combo.clear()
        for label, model_id in self._mt_models_by_pair.get((source_lang, target_lang), []):
            self.mt_model_combo.addItem(label, model_id)

        asr_idx = self.asr_model_combo.findData(prev_asr)
        if asr_idx >= 0:
            self.asr_model_combo.setCurrentIndex(asr_idx)
        mt_idx = self.mt_model_combo.findData(prev_mt)
        if mt_idx >= 0:
            self.mt_model_combo.setCurrentIndex(mt_idx)

    def _on_list_devices(self) -> None:
        config_path = self.config_edit.text().strip() or "grpc_demo/configs/demo.yaml"
        cfg = StreamingWorker._safe_load_config(config_path)
        audio_cfg = cfg.get("audio", {})
        input_cfg = cfg.get("input_mode", {})

        source = FFmpegMicSource(
            sample_rate=int(audio_cfg.get("sample_rate", 16000)),
            chunk_ms=int(audio_cfg.get("transport_chunk_ms", 40)),
            device_hint=self.device_hint_edit.text().strip() or str(input_cfg.get("device_hint", "")),
        )
        devices = source.list_audio_devices()
        if devices:
            text = "\n".join(devices)
            self._append_event(f"[DEVICES] found={len(devices)}")
            QMessageBox.information(self, "Audio Devices", text)
        else:
            self._append_event("[DEVICES] none found")
            QMessageBox.warning(
                self,
                "Audio Devices",
                "No DirectShow microphone devices found. Check FFmpeg and device availability.",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PySide6 GUI client for ASR->MT gRPC demo")
    parser.add_argument("--config", default="grpc_demo/configs/demo.yaml", help="Config YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    window = MainWindow(default_config=args.config)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
