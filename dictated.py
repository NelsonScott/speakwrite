#!/usr/bin/env python3
"""dictated — hold a Whisper model warm; toggle mic dictation via a FIFO.

macOS-style dictation for Linux/Wayland:

    double-tap Ctrl  ->  keyd writes 't' to $XDG_RUNTIME_DIR/dictate.fifo
    first toggle     ->  start streaming the default mic
    while speaking   ->  every natural pause, the phrase so far is transcribed
                         (faster-whisper, GPU) and *typed* into the focused
                         window via ydotool — near-real-time, like macOS
    second toggle    ->  stop; any remaining speech is flushed

Runs as a systemd user service so the model is loaded once at login and
every dictation afterwards starts instantly.

Config via environment (all optional):
    DICTATE_MODEL      faster-whisper model name/path   (default: large-v3)
    DICTATE_DEVICE     cuda | cpu | auto                (default: auto)
    DICTATE_LANG       language code, or 'auto'         (default: en)
    DICTATE_STREAM     1 = type at every pause (default), 0 = only on stop
    DICTATE_PAUSE_MS   silence that commits a phrase    (default: 700)
    DICTATE_KEY_DELAY  ydotool ms/keystroke             (default: 5)
    DICTATE_NO_TYPE    if set, print transcript instead of typing (testing)
    DICTATE_INPUT_CMD  override mic capture command (testing; must emit raw
                       s16le 16kHz mono on stdout)
    YDOTOOL_SOCKET     must match the running ydotoold instance
"""

import os
import shlex
import signal
import stat
import subprocess
import sys
import threading
import time

import numpy as np

RUNTIME_DIR = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
FIFO_PATH = os.path.join(RUNTIME_DIR, "dictate.fifo")

ENGINE = os.environ.get("DICTATE_ENGINE", "whisper")  # whisper | nemotron
NEMO_VENV = os.path.expanduser("~/.local/share/dictate/nemo-venv/bin/python")
NEMO_HELPER = os.path.expanduser("~/.local/share/dictate/nemotron_stream.py")
MODEL_NAME = os.environ.get("DICTATE_MODEL", "large-v3")
DEVICE = os.environ.get("DICTATE_DEVICE", "auto")
LANG = os.environ.get("DICTATE_LANG", "en")
STREAM = os.environ.get("DICTATE_STREAM", "1") != "0"
PAUSE_MS = int(os.environ.get("DICTATE_PAUSE_MS", "450"))
NO_TYPE = bool(os.environ.get("DICTATE_NO_TYPE"))

RATE = 16000
BPS = 2  # bytes/sample, s16le mono
INPUT_CMD = shlex.split(os.environ.get(
    "DICTATE_INPUT_CMD",
    f"parec --rate={RATE} --channels=1 --format=s16le --latency-msec=30",
))

# A phrase with no pause is force-committed after this long (keeps latency
# bounded and the buffer small; a mid-word cut is possible but rare).
MAX_PHRASE_S = 8
# In pure silence, don't let the buffer grow: keep only this much tail.
SILENCE_KEEP_S = 2

# Notification id (-r) so start/stop/result replace one bubble instead of stacking.
NOTIFY_ID = "991199"


def notify(summary, body="", icon="audio-input-microphone", ms=2500):
    subprocess.run(
        ["notify-send", "-r", NOTIFY_ID, "-t", str(ms), "-i", icon, summary, body],
        check=False,
    )


def play(sound):
    """Best-effort UI sound (freedesktop theme ships with GNOME)."""
    path = f"/usr/share/sounds/freedesktop/stereo/{sound}.oga"
    if os.path.exists(path):
        subprocess.Popen(
            ["paplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )


def warmup(model):
    """Transcribe 1s of silence so the first real dictation is instant."""
    list(model.transcribe(np.zeros(RATE, dtype=np.float32))[0])


def load_model():
    """Load on GPU (float16); fall back to a small CPU model if CUDA fails."""
    from faster_whisper import WhisperModel

    if DEVICE in ("auto", "cuda"):
        try:
            m = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")
            warmup(m)  # pay the CUDA kernel-init cost now, not on first dictation
            print(f"loaded {MODEL_NAME} on cuda/float16", flush=True)
            return m
        except Exception as e:  # CUDA/driver mismatch, OOM, ...
            print(f"CUDA load failed ({e!r}); falling back to CPU", flush=True)
            if DEVICE == "cuda":
                raise
    m = WhisperModel("small", device="cpu", compute_type="int8")
    notify("Dictation: GPU unavailable", "Using CPU 'small' model (lower quality).",
           icon="dialog-warning")
    print("loaded small on cpu/int8", flush=True)
    return m


def transcribe(model, audio_f32):
    t0 = time.monotonic()
    segments, info = model.transcribe(
        audio_f32,
        language=None if LANG == "auto" else LANG,
        beam_size=5,
        condition_on_previous_text=False,  # avoids runaway repetition loops
    )
    text = " ".join(
        s.text.strip() for s in segments if s.no_speech_prob < 0.6
    ).strip()
    dt = time.monotonic() - t0
    print(f"transcribed {len(audio_f32)/RATE:.1f}s audio in {dt:.2f}s: {text!r}",
          flush=True)
    return text


# ydotool emits raw US-QWERTY keycodes; unicode punctuation Whisper sometimes
# produces (curly quotes, em-dashes) garbles or stalls it. Normalize to ASCII.
UNICODE_FIXUPS = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": " - ", "…": "...", " ": " ",
})


def type_text(text, raw_delta=False):
    text = text.translate(UNICODE_FIXUPS)
    if not raw_delta:
        # Trailing space so consecutive phrases/dictations join like macOS.
        text = text + " "
    if NO_TYPE:
        print(f"WOULD TYPE: {text!r}", flush=True)
        return
    if not text:
        return
    env = dict(os.environ)
    env.setdefault("YDOTOOL_SOCKET", os.path.join(RUNTIME_DIR, ".ydotool_socket"))
    # 5ms/char: fast but not so fast that Electron apps drop characters.
    subprocess.run(
        ["ydotool", "type", "--key-delay", os.environ.get("DICTATE_KEY_DELAY", "5"),
         "--", text],
        env=env, check=True,
    )


def esc_watcher(stop_evt):
    """While dictating, watch keyd's virtual keyboard for Esc -> cancel.

    Real keyboards are EVIOCGRAB-grabbed by keyd, so we read keyd's own
    output device (readable via a udev uaccess rule). On Esc we poke our
    own FIFO with 'c'; the main loop turns that into a cancel.
    """
    try:
        from evdev import InputDevice, ecodes, list_devices
        import select
        devs = [InputDevice(path) for path in list_devices()]
        devs = [d for d in devs if d.name == "keyd virtual keyboard"]
        if not devs:
            return
        while not stop_evt.is_set():
            r, _, _ = select.select(devs, [], [], 0.2)
            for d in r:
                for ev in d.read():
                    if (ev.type == ecodes.EV_KEY
                            and ev.code == ecodes.KEY_ESC and ev.value == 1):
                        fd = os.open(FIFO_PATH, os.O_WRONLY | os.O_NONBLOCK)
                        os.write(fd, b"c")
                        os.close(fd)
                        return
    except Exception as e:
        print(f"esc watcher unavailable: {e!r}", flush=True)


class Session:
    """One dictation: mic reader thread + VAD/transcribe/type worker thread."""

    def __init__(self, model):
        self.model = model
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.stop_evt = threading.Event()
        self.typed_anything = False
        self.rec = subprocess.Popen(
            INPUT_CMD, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        self.discard = False
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.escwatch = threading.Thread(
            target=esc_watcher, args=(self.stop_evt,), daemon=True)
        self.reader.start()
        self.worker.start()
        self.escwatch.start()

    def _read(self):
        while True:
            data = self.rec.stdout.read(RATE * BPS // 10)  # 100ms
            if not data:
                return
            with self.lock:
                self.buf.extend(data)

    def _take(self, n_bytes):
        """Pop the first n_bytes of the buffer as float32 audio."""
        with self.lock:
            chunk, self.buf = self.buf[:n_bytes], self.buf[n_bytes:]
        return np.frombuffer(bytes(chunk), np.int16).astype(np.float32) / 32768.0

    def _commit(self, n_bytes):
        text = transcribe(self.model, self._take(n_bytes))
        if text:
            try:
                type_text(text)
                self.typed_anything = True
            except Exception as e:
                notify("Typing failed (is ydotoold running?)", str(e),
                       icon="dialog-error")
                print(f"ydotool error: {e!r}", flush=True)

    def _work(self):
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        opts = VadOptions(min_silence_duration_ms=min(PAUSE_MS, 400),
                          speech_pad_ms=150)
        while not self.stop_evt.wait(0.15):
            if not STREAM:
                continue
            try:
                self._poll(opts, get_speech_timestamps)
            except Exception:
                import traceback
                traceback.print_exc()
                sys.stdout.flush()

        # ---- stop: flush whatever remains ----
        self.rec.terminate()
        self.rec.wait(timeout=5)
        self.reader.join(timeout=2)
        with self.lock:
            n = len(self.buf) & ~1
        if n > RATE * BPS // 4:  # anything ≥ 0.25s
            self._commit(n)

    def _poll(self, opts, get_speech_timestamps):
        with self.lock:
            n = len(self.buf) & ~1  # s16 alignment: never split a sample
            audio = np.frombuffer(bytes(self.buf[:n]), np.int16)
        if n < RATE * BPS // 2:  # need at least 0.5s to bother
            return
        ts = get_speech_timestamps(
            audio.astype(np.float32) / 32768.0, opts, sampling_rate=RATE
        )
        if os.environ.get("DICTATE_DEBUG"):
            print(f"poll n={n/(RATE*BPS):.2f}s ts={[(t['start']/RATE, t['end']/RATE) for t in ts]}", flush=True)
        if not ts:
            # pure silence: cap the buffer so hour-long pauses stay cheap
            if n > RATE * BPS * (SILENCE_KEEP_S + 2):
                self._take(n - RATE * BPS * SILENCE_KEEP_S)
            return
        last_end = ts[-1]["end"] * BPS
        trailing_silence_s = (n - last_end) / (RATE * BPS)
        phrase_s = (last_end - ts[0]["start"] * BPS) / (RATE * BPS)
        if trailing_silence_s >= PAUSE_MS / 1000:
            # natural pause: commit everything through the last speech
            self._commit(min(n, last_end + int(0.05 * RATE) * BPS) & ~1)
        elif phrase_s >= MAX_PHRASE_S:
            # no pause for ages: force-commit to keep latency bounded
            self._commit(n)

    def stop(self, discard=False):
        self.discard = discard
        self.stop_evt.set()
        self.worker.join(timeout=30)
        return self.typed_anything


class NemotronHelper:
    """Long-lived streaming STT subprocess (loads once, serves many dictations)."""

    EOT = "\x04"

    def __init__(self):
        self.proc = subprocess.Popen(
            [NEMO_VENV, NEMO_HELPER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None,  # inherit: helper diagnostics land in our journal
        )

    def alive(self):
        return self.proc.poll() is None


class NemotronSession:
    """One dictation on the streaming engine: mic -> helper stdin;
    helper stdout deltas -> typed immediately, word by word."""

    def __init__(self, helper):
        self.helper = helper
        self.discard = False
        self.typed_anything = False
        self.done_evt = threading.Event()
        self.stop_evt = threading.Event()  # for the esc watcher
        self.rec = subprocess.Popen(
            INPUT_CMD, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        self.pump = threading.Thread(target=self._pump, daemon=True)
        self.typer = threading.Thread(target=self._type_deltas, daemon=True)
        self.escwatch = threading.Thread(
            target=esc_watcher, args=(self.stop_evt,), daemon=True)
        self.pump.start()
        self.typer.start()
        self.escwatch.start()

    def _pump(self):
        while True:
            data = self.rec.stdout.read(RATE * BPS // 10)
            if not data:
                return
            try:
                self.helper.proc.stdin.write(data)
                self.helper.proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                return

    def _type_deltas(self):
        for raw in self.helper.proc.stdout:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line == NemotronHelper.EOT:
                self.done_evt.set()
                return
            if line.startswith("+") and not self.discard:
                delta = line[1:]
                if delta:
                    try:
                        type_text(delta, raw_delta=True)
                        self.typed_anything = True
                    except Exception as e:
                        print(f"ydotool error: {e!r}", flush=True)

    def stop(self, discard=False):
        self.discard = discard
        self.stop_evt.set()
        self.rec.terminate()
        self.rec.wait(timeout=5)
        self.pump.join(timeout=3)
        time.sleep(0.4)  # let the helper drain the last audio from the pipe
        try:
            self.helper.proc.send_signal(signal.SIGUSR1)
        except Exception:
            pass
        self.done_evt.wait(timeout=10)
        if self.typed_anything and not discard:
            try:
                type_text("", raw_delta=False)  # trailing space, like macOS
            except Exception:
                pass
        return self.typed_anything


def ensure_fifo():
    if os.path.exists(FIFO_PATH) and not stat.S_ISFIFO(os.stat(FIFO_PATH).st_mode):
        os.unlink(FIFO_PATH)
    if not os.path.exists(FIFO_PATH):
        os.mkfifo(FIFO_PATH, 0o622)  # anyone (i.e. root/keyd) may write


def drain_fifo(fd):
    """Discard toggles queued while we were busy."""
    os.set_blocking(fd, False)
    try:
        while os.read(fd, 4096):
            pass
    except BlockingIOError:
        pass
    os.set_blocking(fd, True)


def main():
    ensure_fifo()
    helper = None
    model = None
    if ENGINE == "nemotron":
        notify("Dictation: loading Nemotron…", "streaming engine", ms=4000)
        helper = NemotronHelper()
    else:
        notify("Dictation: loading model…", MODEL_NAME, ms=4000)
        model = load_model()
    notify("Dictation ready", "Double-tap Ctrl to start/stop.", ms=2000)

    session = None

    # Opening RDWR keeps the FIFO open even with no writers -> reads block
    # instead of spinning on EOF.
    fd = os.open(FIFO_PATH, os.O_RDWR)

    while True:
        data = os.read(fd, 4096)
        if not data:
            continue

        if b"c" in data:  # Esc pressed -> cancel, type nothing further
            if session is not None:
                session.stop(discard=True)
                session = None
                play("dialog-warning")
                notify("Dictation cancelled", "Discarded (Esc).", ms=1500)
                print("cancelled via Esc", flush=True)
                drain_fifo(fd)
            continue

        if session is None:
            if ENGINE == "nemotron":
                if not helper.alive():
                    notify("Dictation: restarting Nemotron…", ms=3000)
                    helper = NemotronHelper()
                session = NemotronSession(helper)
            else:
                session = Session(model)
            play("message-new-instant")
            notify("🎤 Dictating…",
                   "Text appears at each pause. Double-tap Ctrl to stop.",
                   ms=3600000)
        else:
            typed = session.stop()
            session = None
            play("message")
            notify("Dictation off",
                   "" if typed else "No speech detected.", ms=1500)
            drain_fifo(fd)


if __name__ == "__main__":
    sys.exit(main())
