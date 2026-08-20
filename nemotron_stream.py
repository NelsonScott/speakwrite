#!/usr/bin/env python3
"""Nemotron cache-aware streaming STT helper (runs in the NeMo py3.12 venv).

Long-lived: spawned once by dictated.py at startup (model load ~6s), then
serves many dictations.

Protocol:
  stdin   raw s16le 16 kHz mono audio (only flows while a dictation is live)
  SIGUSR1 end of dictation: flush the tail, emit EOT, reset decoder state
  stdout  incremental transcript deltas, one line each, prefixed "+" —
          append-only by construction (RNNT never revises emitted tokens);
          a line containing only "\\x04" marks end-of-dictation (EOT)
  stderr  diagnostics
"""

import os
import queue
import signal
import sys
import threading
import time

import numpy as np
import torch

MODEL_NAME = os.environ.get("NEMOTRON_MODEL", "nvidia/nemotron-speech-streaming-en-0.6b")
RATE = 16000
STEP_MS = int(os.environ.get("NEMOTRON_STEP_MS", "160"))
STEP_BYTES = RATE * 2 * STEP_MS // 1000
# right-attention context (model card: [70,1]=160ms ... [70,13]=1.12s lookahead)
RIGHT_CTX = int(os.environ.get("NEMOTRON_RIGHT_CONTEXT", "1"))
HOLD = 2   # right-edge mel frames shift as audio grows; hold them back
EOT = "\x04"


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def emit(text):
    # newlines inside a delta would break the line protocol
    print("+" + text.replace("\n", " "), flush=True)


def extract_text(hyps):
    h = hyps[0]
    return (h.text if hasattr(h, "text") else h) or ""


class Decoder:
    """Wraps one dictation's streaming state; reset() starts the next one."""

    def __init__(self, model, sbuffer):
        self.model = model
        self.sbuffer = sbuffer
        self.reset(emit_deltas=False)

    def reset(self, emit_deltas=True):
        (self.cache_ch, self.cache_t, self.cache_len) = \
            self.model.encoder.get_initial_cache_state(batch_size=1)
        self.prev_hyp = None
        self.pred_out = None
        self.step = 0
        self.emitted = ""
        self.all_audio = np.zeros(0, dtype=np.float32)
        self.frames_sent = 0
        self.stream_id = -1
        self.emit_deltas = emit_deltas
        self.sbuffer.reset_buffer()

    def feed(self, raw_bytes, final=False):
        if raw_bytes:
            samples = np.frombuffer(raw_bytes, np.int16).astype(np.float32) / 32768.0
            self.all_audio = np.concatenate([self.all_audio, samples])
        if self.all_audio.size < 400:  # under one mel window: nothing to do
            return
        # Featurize the WHOLE dictation so far (cheap on GPU) and append only
        # NEW frames: per-snippet featurization glitches at chunk boundaries.
        feats, flen = self.sbuffer.preprocess_audio(self.all_audio)
        usable = int(flen) - (0 if final else HOLD)
        if usable > self.frames_sent:
            new = feats[:, :, self.frames_sent:usable]
            self.frames_sent = usable
            _, _, sid = self.sbuffer.append_processed_signal(new, stream_id=self.stream_id)
            self.stream_id = 0 if sid is None or sid < 0 else sid
        for chunk_audio, chunk_lengths in self.sbuffer:
            with torch.inference_mode():
                (
                    self.pred_out, texts, self.cache_ch, self.cache_t,
                    self.cache_len, self.prev_hyp,
                ) = self.model.conformer_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=self.cache_ch,
                    cache_last_time=self.cache_t,
                    cache_last_channel_len=self.cache_len,
                    keep_all_outputs=final and self.sbuffer.is_buffer_empty(),
                    previous_hypotheses=self.prev_hyp,
                    previous_pred_out=self.pred_out,
                    drop_extra_pre_encoded=(
                        0 if self.step == 0
                        else self.model.encoder.streaming_cfg.drop_extra_pre_encoded
                    ),
                    return_transcription=True,
                )
            self.step += 1
            full = extract_text(texts)
            if len(full) > len(self.emitted):
                if self.emit_deltas:
                    emit(full[len(self.emitted):])
                self.emitted = full


def main():
    t0 = time.monotonic()
    from nemo.utils import logging as nlog
    nlog.setLevel(nlog.ERROR)
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.utils.streaming_utils import (
        CacheAwareStreamingAudioBuffer,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME, map_location=device)
    model = model.to(device).eval()
    if hasattr(model.encoder, "set_default_att_context_size"):
        try:
            model.encoder.set_default_att_context_size([70, RIGHT_CTX])
            model.encoder.setup_streaming_params()
        except Exception as e:
            log(f"att context setup skipped: {e!r}")

    online_norm = getattr(model.cfg.preprocessor, "normalize", None) in (
        "per_feature", "all_feature",
    )
    sbuffer = CacheAwareStreamingAudioBuffer(
        model=model, online_normalization=online_norm, pad_and_drop_preencoded=False
    )
    dec = Decoder(model, sbuffer)
    dec.feed(b"\x00\x00" * RATE, final=True)  # warm up CUDA kernels on 1s silence
    dec.reset()
    scfg = model.encoder.streaming_cfg
    log(f"ready in {time.monotonic() - t0:.1f}s on {device} "
        f"(chunk={scfg.chunk_size}, shift={scfg.shift_size}, norm={online_norm})")

    flush_req = threading.Event()
    signal.signal(signal.SIGUSR1, lambda *_: flush_req.set())

    q = queue.Queue()

    def reader():
        while True:
            d = os.read(0, STEP_BYTES)
            q.put(d)
            if not d:
                return

    threading.Thread(target=reader, daemon=True).start()

    pending = b""
    eof = False
    while not eof:
        try:
            d = q.get(timeout=0.05)
            if not d:
                eof = True
            else:
                pending += d
        except queue.Empty:
            pass

        if flush_req.is_set() and q.empty():
            # drain anything that raced in between the check and now
            try:
                while True:
                    d = q.get_nowait()
                    if d:
                        pending += d
                    else:
                        eof = True
            except queue.Empty:
                pass
            take = len(pending) & ~1
            dec.feed(pending[:take], final=True)
            pending = b""
            print(EOT, flush=True)
            log(f"dictation done: {dec.step} steps, {len(dec.emitted)} chars")
            dec.reset()
            flush_req.clear()
            continue

        while len(pending) >= STEP_BYTES:
            t_hop = time.monotonic()
            dec.feed(pending[:STEP_BYTES])
            hop_ms = (time.monotonic() - t_hop) * 1000
            if hop_ms > 400:
                log(f"SLOW HOP: {hop_ms:.0f}ms at step {dec.step} "
                    f"({dec.all_audio.size / RATE:.1f}s into dictation) — "
                    f"likely GPU contention (ollama/game?)")
            pending = pending[STEP_BYTES:]

    # stdin closed: daemon is going away
    take = len(pending) & ~1
    dec.feed(pending[:take], final=True)
    print(EOT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
