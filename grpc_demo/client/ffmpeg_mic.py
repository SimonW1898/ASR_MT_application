"""FFmpeg microphone capture adapter."""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional


@dataclass(frozen=True)
class MicFrame:
    """Single microphone frame in PCM16 format."""

    pcm16: bytes
    t0: int
    t1: int
    frame_idx: int


class FFmpegMicSource:
    """Capture microphone audio from FFmpeg and stream PCM16 frames."""

    def __init__(
        self,
        sample_rate: int,
        chunk_ms: int,
        device_hint: str = "",
        ffmpeg_bin: str = "ffmpeg",
    ):
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.device_hint = device_hint
        self.ffmpeg_bin = ffmpeg_bin

        self.frame_samples = int(sample_rate * chunk_ms / 1000)
        self.frame_bytes = self.frame_samples * 2

    def list_audio_devices(self) -> list[str]:
        """List DirectShow audio input devices available via FFmpeg."""
        cmd = [self.ffmpeg_bin, "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return []

        output = result.stderr
        devices: list[str] = []
        for line in output.splitlines():
            if "Alternative name" in line:
                continue
            if "(audio)" in line:
                match = re.search(r'"([^"]+)"', line)
                if match:
                    name = match.group(1)
                    if name not in devices:
                        devices.append(name)
        return devices

    def frames(self, stop_requested: Optional[Callable[[], bool]] = None) -> Iterator[MicFrame]:
        """Yield PCM16 frames from microphone until stopped."""
        devices = self.list_audio_devices()
        selected = self._select_device(devices)
        cmd = self._build_cmd(selected)

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("FFmpeg executable not found on PATH.") from exc

        try:
            time.sleep(0.4)
            if process.poll() is not None:
                err = b""
                if process.stderr is not None:
                    err = process.stderr.read()
                stderr_text = err.decode("utf-8", errors="ignore")
                raise RuntimeError(f"FFmpeg failed to start microphone capture: {stderr_text}")

            frame_idx = 0
            sample_cursor = 0

            while True:
                if stop_requested is not None and stop_requested():
                    break

                if process.stdout is None:
                    break
                raw = process.stdout.read(self.frame_bytes)
                if not raw or len(raw) < 2:
                    break

                n_samples = len(raw) // 2
                t0 = sample_cursor
                t1 = sample_cursor + n_samples
                sample_cursor = t1

                yield MicFrame(pcm16=raw, t0=t0, t1=t1, frame_idx=frame_idx)
                frame_idx += 1
        finally:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

    def _select_device(self, devices: list[str]) -> str:
        """Select capture device by hint, falling back to first detected."""
        hint = (self.device_hint or "").strip()
        if hint:
            if hint in devices:
                return hint
            for device in devices:
                if hint.lower() in device.lower():
                    return device
            return hint
        if devices:
            return devices[0]
        return "default"

    def _build_cmd(self, device_name: str) -> list[str]:
        """Build FFmpeg command for DirectShow mic capture."""
        return [
            self.ffmpeg_bin,
            "-loglevel",
            "error",
            "-f",
            "dshow",
            "-i",
            f"audio={device_name}",
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-f",
            "s16le",
            "pipe:1",
        ]
