"""
Real-time streaming translation pipeline with two threads.

Thread 1 (Listener): Captures audio via FFmpeg -> applies emit policy -> enqueues segments
Thread 2 (Processor): Dequeues segments -> ASR -> Merge -> MT -> Output
"""

import time
import threading
import queue
import subprocess
import re
from dataclasses import dataclass
from typing import Optional, List, Callable, Tuple

import numpy as np

from .core.types import DEFAULT_SR, DEFAULT_FRAME_SAMPLES
from .sources.gate import RMSGate
from .preprocess import clean_audio, PreprocessConfig
from .buffers.buffers import (
    WaitKSlidingWindowBuffer,
    FixedWindowBuffer,
    SilenceTerminatedBuffer,
    VADBuffer,
)
from .buffers.merge import merge_transcripts_align, AlignMergeConfig
from .asr import WhisperASR
from .mt import MarianMT
from .sinks import RealtimeSink, RealtimeConsoleSink


@dataclass
class RealtimeConfig:
    """Configuration for real-time translation pipeline."""
    device: Optional[str] = None
    sample_rate: int = DEFAULT_SR
    frame_samples: int = DEFAULT_FRAME_SAMPLES

    policy: str = "silence_terminated"

    silence_ms: int = 500
    min_ms: int = 400
    max_ms: int = 8000
    speech_threshold: float = 0.01

    pre_ms: int = 200
    post_ms: int = 200

    win_ms: int = 2000
    hop_ms: int = 500
    k_ms: int = 2000

    max_queue_size: int = 10000

    asr_model_id: str = "openai/whisper-small"
    mt_model_id: str = "Helsinki-NLP/opus-mt-en-de"
    offline: bool = False

    preprocess_mode: str = "bandpass"  # "none", "bandpass", "ml"
    preprocess_ml_model: str = "noisereduce"
    preprocess_ml_device: str = "cpu"
    preprocess_config: Optional[PreprocessConfig] = None

    merge_enabled: bool = True
    merge_cfg: Optional[AlignMergeConfig] = None
    mt_mode: str = "segment"  # "segment" or "merged"


@dataclass
class AudioSegment:
    """Audio segment ready for ASR+MT processing."""
    audio: np.ndarray
    start_sample: int
    end_sample: int
    timestamp: float
    reason: str
    first_speech_time: Optional[float] = None


@dataclass
class TranslationResult:
    """Result from processing an audio segment."""
    segment_idx: int
    asr_text: str
    mt_text: str
    merged_asr: str
    merged_mt: str
    audio_time_s: float
    queue_latency_s: float
    process_time_s: float
    reason: str
    first_word_to_translation_s: Optional[float] = None


def list_audio_devices() -> List[str]:
    """List available audio input devices via FFmpeg on Windows."""
    cmd = "ffmpeg -list_devices true -f dshow -i dummy"

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stderr
    except subprocess.TimeoutExpired:
        return []
    except Exception:
        return []

    devices = []
    for line in output.split("\n"):
        if "Alternative name" in line:
            continue
        if "(audio)" in line:
            match = re.search(r'"([^"]+)"', line)
            if match:
                device_name = match.group(1)
                if device_name not in devices:
                    devices.append(device_name)

    return devices


def select_audio_device(preferred: Optional[str] = None) -> str:
    """Select an audio device with auto-detection fallback."""
    devices = list_audio_devices()

    if not devices:
        print("[WARN] No audio devices found. Using 'default'.")
        return "default"

    print(f"\n[INFO] Found {len(devices)} audio device(s):")
    for i, dev in enumerate(devices):
        print(f"  [{i}] {dev}")

    if preferred:
        if preferred in devices:
            print(f"\n[INFO] Using specified device: {preferred}")
            return preferred
        for dev in devices:
            if preferred.lower() in dev.lower():
                print(f"\n[INFO] Using matched device: {dev}")
                return dev
        print(f"\n[WARN] Device '{preferred}' not found, using first available.")

    print(f"\n[INFO] Auto-selected device: {devices[0]}")
    return devices[0]


class AudioListener:
    """Capture audio via FFmpeg and emit segments based on buffer policy."""

    def __init__(
        self,
        config: RealtimeConfig,
        segment_queue: queue.Queue,
        preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.config = config
        self.segment_queue = segment_queue
        self.preprocess_fn = preprocess_fn or (lambda x: x)

        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.process: Optional[subprocess.Popen] = None

        self.buffer = self._create_buffer()

        self.frames_read = 0
        self.segments_emitted = 0
        self.bytes_per_sample = 2

        self.first_speech_time: Optional[float] = None

    def _create_buffer(self):
        gate = RMSGate(threshold=self.config.speech_threshold)
        sr = self.config.sample_rate

        if self.config.policy == "fixed_window":
            return FixedWindowBuffer(sr=sr, win_s=self.config.win_ms / 1000, gate=gate)
        if self.config.policy == "silence_terminated":
            return SilenceTerminatedBuffer(
                sr=sr,
                min_s=self.config.min_ms / 1000,
                max_s=self.config.max_ms / 1000,
                silence_s=self.config.silence_ms / 1000,
                gate=gate,
            )
        if self.config.policy == "wait_k_sliding":
            return WaitKSlidingWindowBuffer(
                sr=sr,
                k_s=self.config.k_ms / 1000,
                hop_s=self.config.hop_ms / 1000,
                win_s=self.config.win_ms / 1000,
            )
        if self.config.policy == "vad":
            return VADBuffer(
                sr=sr,
                pre_s=self.config.pre_ms / 1000,
                post_s=self.config.post_ms / 1000,
                min_speech_s=self.config.min_ms / 1000,
                end_silence_s=self.config.silence_ms / 1000,
                max_s=self.config.max_ms / 1000,
                gate=gate,
            )
        raise ValueError(f"Unknown policy: {self.config.policy}")

    def _build_ffmpeg_cmd(self, device: str) -> str:
        sr = self.config.sample_rate
        return (
            f"ffmpeg -loglevel error -f dshow -i audio=\"{device}\" "
            f"-ac 1 -ar {sr} -f s16le pipe:1"
        )

    def _start_ffmpeg(self, device: str) -> subprocess.Popen:
        cmd = self._build_ffmpeg_cmd(device)
        print(f"[LISTENER] Starting FFmpeg: {cmd}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )

        time.sleep(0.5)

        if process.poll() is not None:
            stderr = process.stderr.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"FFmpeg failed to start: {stderr}")

        return process

    def _listen_loop(self, device: str):
        try:
            self.process = self._start_ffmpeg(device)
            print("[LISTENER] Audio capture started. Listening...")

            bytes_per_frame = self.config.frame_samples * self.bytes_per_sample
            sample_counter = 0

            while self.running:
                raw = self.process.stdout.read(bytes_per_frame)

                if len(raw) == 0:
                    if self.process.poll() is not None:
                        stderr = self.process.stderr.read().decode("utf-8", errors="ignore")
                        print(f"[LISTENER] FFmpeg terminated: {stderr}")
                    break

                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                audio = self.preprocess_fn(audio)

                t0 = sample_counter
                t1 = sample_counter + len(audio)
                sample_counter = t1
                self.frames_read += 1

                segments = self.buffer.push(audio, t0, t1)

                if segments and self.first_speech_time is None:
                    self.first_speech_time = time.time()

                for seg in segments:
                    audio_segment = AudioSegment(
                        audio=seg.audio,
                        start_sample=seg.start_sample,
                        end_sample=seg.end_sample,
                        timestamp=time.time(),
                        reason=seg.reason,
                        first_speech_time=self.first_speech_time,
                    )
                    self.segment_queue.put(audio_segment)
                    self.segments_emitted += 1

            if self.running:
                flush_segments = self.buffer.flush(sample_counter)
                for seg in flush_segments:
                    audio_segment = AudioSegment(
                        audio=seg.audio,
                        start_sample=seg.start_sample,
                        end_sample=seg.end_sample,
                        timestamp=time.time(),
                        reason="flush",
                    )
                    self.segment_queue.put(audio_segment)
                    self.segments_emitted += 1

        except Exception as e:
            print(f"[LISTENER] Error: {e}")

        finally:
            if self.process:
                self.process.terminate()
                self.process.wait()
            print(f"[LISTENER] Stopped. Frames: {self.frames_read}, Segments: {self.segments_emitted}")

    def start(self, device: str):
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, args=(device,), daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()
        if self.thread:
            self.thread.join(timeout=2.0)


class TranslationProcessor:
    """Process audio segments through ASR -> Merge -> MT pipeline."""

    def __init__(
        self,
        config: RealtimeConfig,
        segment_queue: queue.Queue,
        sink: Optional[RealtimeSink] = None,
        result_callback: Optional[Callable[[TranslationResult], None]] = None,
    ):
        self.config = config
        self.segment_queue = segment_queue
        self.sink = sink or RealtimeConsoleSink()
        self.result_callback = result_callback

        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.asr_model = None
        self.mt_model = None

        self.merge_cfg = config.merge_cfg or AlignMergeConfig(
            tail_tokens=50,
            head_tokens=50,
            min_match_tokens=2,
            max_search_shift=8,
        )

        self.merged_asr = ""
        self.merged_mt = ""
        self.segments_processed = 0

        self.first_speech_time: Optional[float] = None
        self.first_translation_time: Optional[float] = None

    def _load_models(self):
        print("\n" + "=" * 60)
        print("[PROCESSOR] Loading models...")
        print("=" * 60)

        print(f"[1/2] Loading Whisper ASR ({self.config.asr_model_id})...")
        self.asr_model = WhisperASR(
            model_id=self.config.asr_model_id,
            offline=self.config.offline,
        )
        print("      ok ASR ready")

        print(f"[2/2] Loading MarianMT ({self.config.mt_model_id})...")
        self.mt_model = MarianMT(
            model_id=self.config.mt_model_id,
            offline=self.config.offline,
        )
        print("      ok MT ready")

        print("=" * 60)
        print("[PROCESSOR] Models loaded")
        print("=" * 60 + "\n")

    def _process_loop(self):
        if self.asr_model is None:
            self._load_models()

        print("[PROCESSOR] Ready. Waiting for audio segments...")

        while self.running:
            try:
                segment = self.segment_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                result = self._process_segment(segment)

                self.sink.on_result(result)

                if self.result_callback:
                    self.result_callback(result)

            except Exception as e:
                print(f"[PROCESSOR] Error processing segment: {e}")

            finally:
                self.segment_queue.task_done()

        print(f"[PROCESSOR] Stopped. Segments processed: {self.segments_processed}")

    def _process_segment(self, segment: AudioSegment) -> TranslationResult:
        process_start = time.time()

        asr_result = self.asr_model.transcribe(segment.audio, sr=self.config.sample_rate)
        asr_text = asr_result.text if hasattr(asr_result, "text") else str(asr_result)

        if self.config.merge_enabled:
            if self.merged_asr:
                self.merged_asr = merge_transcripts_align(self.merged_asr, asr_text, self.merge_cfg)
            else:
                self.merged_asr = asr_text.strip()
        else:
            if self.merged_asr:
                self.merged_asr = self.merged_asr.strip() + " " + asr_text.strip()
            else:
                self.merged_asr = asr_text.strip()

        mt_text = ""
        if self.config.mt_mode == "merged":
            mt_text = self.mt_model.translate(self.merged_asr) if self.merged_asr else ""
            self.merged_mt = mt_text
        else:
            mt_text = self.mt_model.translate(asr_text) if asr_text else ""
            if self.merged_mt:
                self.merged_mt = self.merged_mt.strip() + " " + mt_text.strip()
            else:
                self.merged_mt = mt_text.strip()

        process_time = time.time() - process_start
        self.segments_processed += 1

        translation_complete_time = time.time()
        if self.first_speech_time is None and segment.first_speech_time is not None:
            self.first_speech_time = segment.first_speech_time
        if self.first_translation_time is None and self.merged_mt:
            self.first_translation_time = translation_complete_time

        first_word_to_translation = None
        if self.first_speech_time is not None:
            first_word_to_translation = translation_complete_time - self.first_speech_time

        return TranslationResult(
            segment_idx=self.segments_processed,
            asr_text=asr_text,
            mt_text=mt_text if self.config.mt_mode == "segment" else self.merged_mt,
            merged_asr=self.merged_asr,
            merged_mt=self.merged_mt,
            audio_time_s=segment.end_sample / self.config.sample_rate,
            queue_latency_s=time.time() - segment.timestamp,
            process_time_s=process_time,
            reason=segment.reason,
            first_word_to_translation_s=first_word_to_translation,
        )

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)

    def get_final_output(self) -> Tuple[str, str]:
        return self.merged_asr, self.merged_mt

    def reset(self):
        self.merged_asr = ""
        self.merged_mt = ""
        self.segments_processed = 0
        self.first_speech_time = None
        self.first_translation_time = None

    def get_timing_stats(self) -> dict:
        stats = {
            "segments_processed": self.segments_processed,
            "first_speech_time": self.first_speech_time,
            "first_translation_time": self.first_translation_time,
        }
        if self.first_speech_time and self.first_translation_time:
            stats["first_word_to_first_translation_s"] = (
                self.first_translation_time - self.first_speech_time
            )
        return stats


class RealtimeTranslationPipeline:
    """Orchestrates real-time translation with two threads."""

    def __init__(
        self,
        config: RealtimeConfig,
        sink: Optional[RealtimeSink] = None,
        result_callback: Optional[Callable[[TranslationResult], None]] = None,
    ):
        self.config = config
        self.sink = sink or RealtimeConsoleSink()

        self.segment_queue = queue.Queue(maxsize=config.max_queue_size)

        self.listener = AudioListener(
            config=config,
            segment_queue=self.segment_queue,
            preprocess_fn=self._preprocess,
        )
        self.processor = TranslationProcessor(
            config=config,
            segment_queue=self.segment_queue,
            sink=self.sink,
            result_callback=result_callback,
        )

    def _preprocess(self, audio: np.ndarray) -> np.ndarray:
        mode = self.config.preprocess_mode

        if mode == "none":
            return audio

        if self.config.preprocess_config is not None:
            preprocess_cfg = self.config.preprocess_config
        else:
            preprocess_cfg = PreprocessConfig(
                mode=mode,
                ml_model=self.config.preprocess_ml_model,
                ml_device=self.config.preprocess_ml_device,
            )

        try:
            cleaned, _stats = clean_audio(
                audio,
                sr=self.config.sample_rate,
                mode=mode,
                config=preprocess_cfg,
            )
            return cleaned
        except ImportError as e:
            if not hasattr(self, "_preprocess_warned"):
                print(f"[WARN] Preprocessing failed: {e}")
                print("[WARN] Falling back to no preprocessing.")
                self._preprocess_warned = True
            return audio
        except Exception as e:
            if not hasattr(self, "_preprocess_warned"):
                print(f"[WARN] Preprocessing error: {e}")
                self._preprocess_warned = True
            return audio

    def run(self, device: Optional[str] = None):
        selected_device = select_audio_device(device or self.config.device)

        print("\n" + "=" * 70)
        print("  REAL-TIME STREAMING TRANSLATION")
        print("=" * 70)
        print(f"  Device:  {selected_device}")
        print(f"  Policy:  {self.config.policy}")
        print(f"  ASR:     {self.config.asr_model_id}")
        print(f"  MT:      {self.config.mt_model_id}")
        print("=" * 70)
        print("  Press Ctrl+C to stop")
        print("=" * 70 + "\n")

        try:
            self.processor.start()
            time.sleep(0.5)

            self.listener.start(selected_device)

            while self.listener.running and self.processor.running:
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\n[MAIN] Keyboard interrupt received. Stopping...")

        finally:
            self.listener.stop()

            try:
                self.segment_queue.join()
            except Exception:
                pass

            self.processor.stop()

            final_asr, final_mt = self.processor.get_final_output()
            timing_stats = self.processor.get_timing_stats()

            self.sink.on_final(final_asr, final_mt, timing_stats)
            self.sink.close()

    def get_final_output(self) -> Tuple[str, str]:
        return self.processor.get_final_output()


__all__ = [
    "RealtimeConfig",
    "AudioSegment",
    "TranslationResult",
    "AudioListener",
    "TranslationProcessor",
    "RealtimeTranslationPipeline",
    "list_audio_devices",
    "select_audio_device",
]
