"""gRPC server entry point for ASR->MT demo."""

from __future__ import annotations

import argparse
import logging
import queue
import time
from concurrent import futures
from typing import Iterator

import grpc
import yaml

try:
    from grpc_demo.proto_gen import asrmt_pb2, asrmt_pb2_grpc
except Exception as exc:  # pragma: no cover - import guard for missing generation
    raise RuntimeError(
        "Missing generated proto code. Generate grpc_demo/proto_gen/*.py first."
    ) from exc

from .audio_ingress import GrpcAudioIngress
from .logger import JsonlLogger
from .model_manager import ModelManager
from .processor import ProcessorResult, TranslationProcessor
from .session import SessionConfig, SessionState


_LOGGER = logging.getLogger(__name__)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 50051


class ASRMTService(asrmt_pb2_grpc.ASRMTServiceServicer):
    """Server with segmentation and ASR/MT processing."""

    def StreamSession(
        self,
        request_iterator: Iterator[asrmt_pb2.ClientMessage],
        context: grpc.ServicerContext,
    ) -> Iterator[asrmt_pb2.ServerMessage]:
        try:
            first = next(request_iterator)
        except StopIteration:
            yield _status("ERROR", "No config message received")
            return

        if not first.HasField("config"):
            yield _status("ERROR", "First message must be config")
            return

        cfg = SessionConfig.from_proto(first.config)
        _LOGGER.info("Session config received: %s", cfg)

        yield _status("LOADING", "Initializing session")
        state = SessionState(cfg)
        ingress = GrpcAudioIngress(cfg)
        result_queue: queue.Queue = queue.Queue()
        model_manager = ModelManager(cfg)
        processor = TranslationProcessor(cfg, result_queue, model_manager=model_manager)
        logger = JsonlLogger(cfg.runtime.log_dir)
        processor.start()
        state.mark_ready()
        yield _status("READY", "Server ready")

        for msg in request_iterator:
            if msg.HasField("control"):
                action = msg.control.action.upper().strip()
                if action == "STOP":
                    if state.last_t1 is not None:
                        for seg in ingress.flush(state.last_t1):
                            processor.submit(seg, time.time())
                    processor.stop()
                    yield from _drain_results(result_queue, logger)
                    _LOGGER.info(
                        "Session stopped. Frames=%s Bytes=%s Samples=%s",
                        ingress.stats.frames_received,
                        ingress.stats.bytes_received,
                        ingress.stats.samples_received,
                    )
                    logger.close()
                    yield _status("STOPPED", "Session stopped")
                    return
                if action == "FLUSH":
                    if state.last_t1 is not None:
                        for seg in ingress.flush(state.last_t1):
                            processor.submit(seg, time.time())
                        yield from _drain_results(result_queue, logger)
                    continue
            if msg.HasField("audio"):
                if not state.ready:
                    continue
                state.update_last_t1(msg.audio.t1)
                segments = ingress.push_pcm16(msg.audio.pcm16, msg.audio.t0, msg.audio.t1)
                for seg in segments:
                    processor.submit(seg, time.time())
                yield from _drain_results(result_queue, logger)
                continue

    

def _status(state: str, message: str) -> asrmt_pb2.ServerMessage:
    return asrmt_pb2.ServerMessage(status=asrmt_pb2.Status(state=state, message=message))


def _drain_results(
    result_queue: queue.Queue,
    logger: JsonlLogger,
) -> Iterator[asrmt_pb2.ServerMessage]:
    while True:
        try:
            result: ProcessorResult = result_queue.get_nowait()
        except queue.Empty:
            break

        logger.write_result(result)
        yield asrmt_pb2.ServerMessage(
            segment=asrmt_pb2.SegmentResult(
                segment_id=result.segment_id,
                asr_chunk_text=result.asr_chunk_text,
                mt_chunk_text=result.mt_chunk_text,
                merged_asr=result.merged_asr,
                merged_mt=result.merged_mt,
                reason=result.reason,
                audio_time_s=result.audio_time_s,
                queue_latency_ms=result.queue_latency_ms,
                process_time_ms=result.process_time_ms,
                e2e_ms=result.e2e_ms or 0.0,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASR->MT gRPC server")
    parser.add_argument(
        "--config",
        default="grpc_demo/configs/demo.yaml",
        help="Config YAML path",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind host (overrides config)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port (overrides config)")
    parser.add_argument("--max-workers", type=int, default=4, help="gRPC worker threads")
    return parser.parse_args()


def _load_runtime_from_config(path: str, default_host: str, default_port: int) -> tuple[str, int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError):
        return default_host, default_port

    runtime = cfg.get("runtime", {})
    host = str(runtime.get("grpc_host", default_host))
    try:
        port = int(runtime.get("grpc_port", default_port))
    except (TypeError, ValueError):
        port = default_port
    return host, port


def serve() -> None:
    args = parse_args()
    cfg_host, cfg_port = _load_runtime_from_config(
        args.config,
        default_host=args.host,
        default_port=args.port,
    )
    host = cfg_host if args.host == DEFAULT_HOST else args.host
    port = cfg_port if args.port == DEFAULT_PORT else args.port

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_workers))
    asrmt_pb2_grpc.add_ASRMTServiceServicer_to_server(ASRMTService(), server)

    address = f"{host}:{port}"
    server.add_insecure_port(address)
    _LOGGER.info("gRPC server listening on %s", address)

    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
