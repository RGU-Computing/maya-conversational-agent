from __future__ import annotations

import os
import sys
import time
import queue
import logging as log
from math import isfinite
import types
from dataclasses import dataclass
from io import BytesIO
from threading import Thread, Event, Lock
from typing import Optional, Any
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from maya.framework.conversation.personalization import SessionConfig
from maya.framework.integrations.speech import init_asr, init_tts, transcribe

import Ice  # from zeroc-ice


def _install_ice_compat() -> None:
    # -----------------------------
    # 1) Ice.loadSlice: new Ice expects list[str]
    # -----------------------------
    orig_loadSlice = Ice.loadSlice

    def loadSlice_compat(cmd, *args, **kwargs):
        if isinstance(cmd, str):
            incs = []
            for d in ("/usr/share/slice", "/usr/share/slice/phyxio"):
                if os.path.isdir(d):
                    incs.append(f"-I{d}")
            return orig_loadSlice([*incs, cmd], *args, **kwargs)
        return orig_loadSlice(cmd, *args, **kwargs)

    Ice.loadSlice = loadSlice_compat  # type: ignore

    # -----------------------------
    # 2) Communicator.stringToIdentity: old code calls communicator.stringToIdentity(...)
    #    new Ice provides Ice.stringToIdentity(...)
    # -----------------------------
    def _ensure_stringToIdentity(communicator) -> None:
        if hasattr(communicator, "stringToIdentity"):
            return
        if not hasattr(Ice, "stringToIdentity"):
            return

        def _sti(self, s: str):
            return Ice.stringToIdentity(s)

        # Try patching the class (best: keeps type == Communicator)
        try:
            setattr(communicator.__class__, "stringToIdentity", _sti)
            return
        except Exception:
            pass

        # Fallback: patch instance (may fail if Communicator has no __dict__)
        try:
            communicator.stringToIdentity = types.MethodType(_sti, communicator)  # type: ignore
        except Exception:
            pass

    # -----------------------------
    # 3) Ice.initialize: old code uses data=..., new Ice expects initData=...
    # -----------------------------
    orig_initialize = Ice.initialize

    def initialize_compat(*args, **kwargs):
        if "data" in kwargs and "initData" not in kwargs:
            data = kwargs.pop("data")
            try:
                comm = orig_initialize(*args, initData=data, **kwargs)
            except TypeError:
                comm = orig_initialize(*args, **kwargs)
        else:
            comm = orig_initialize(*args, **kwargs)

        _ensure_stringToIdentity(comm)
        return comm

    Ice.initialize = initialize_compat  # type: ignore

# _install_ice_compat()


from phyxio import connect

load_dotenv()
log.basicConfig(level=log.INFO)


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class VoiceConfig:
    max_seconds: float = float(os.getenv("MAYA_VOICE_SECONDS", "20"))
    rate: int = int(os.getenv("MAYA_VOICE_SAMPLE_RATE", "16000"))
    silence_seconds: float = float(os.getenv("MAYA_VOICE_SILENCE_SECONDS", "1.5"))
    silence_threshold: float = float(os.getenv("MAYA_VOICE_SILENCE_THRESHOLD", "0.015"))
    min_seconds: float = float(os.getenv("MAYA_VOICE_MIN_SECONDS", "0.6"))
    poll_seconds: float = float(os.getenv("MAYA_VOICE_POLL_SECONDS", "0.1"))
    error_text: str = os.getenv("MAYA_VOICE_ERROR_TEXT", "Voice input is temporarily unavailable. Please tap to try again.")
    mic_device: Optional[int] = None  # MAYA_MIC_DEVICE=int index

    def __post_init__(self):
        if any(not isfinite(value) or value <= 0 for value in (self.max_seconds, self.poll_seconds)):
            raise ValueError("Voice duration and polling interval must be finite and positive.")


def _mic_device_from_env() -> Optional[int]:
    v = os.getenv("MAYA_MIC_DEVICE")
    return int(v) if v and v.isdigit() else None


def _list_audio_devices_once() -> None:
    try:
        devs = sd.query_devices()
        log.info("---- Audio Devices ----")
        for i, d in enumerate(devs):
            log.info("%d | %s | in=%s out=%s", i, d["name"], d["max_input_channels"], d["max_output_channels"])
        log.info("sd.default.device=%s", sd.default.device)
        log.info("-----------------------")
    except Exception as e:
        log.warning("Could not list audio devices: %s", e)


# -----------------------------
# Phyxio Connector
# -----------------------------
class MyPhyxioConnector(connect.PhyxioConnector):
    """Delegate documented callbacks; leave the connector's internal methods alone."""

    def __init__(self, service: PhyxioService):
        self._maya_service = service
        service._phyxio = self  # Available before the native connection thread starts.
        super().__init__()

    def on_client_connected(self):
        super().on_client_connected()
        self._maya_service.client_connected = True
        log.info("Phyxio Mirror connected")

    def change_interaction_state_request(self, active: bool):
        return self._maya_service._handle_interaction_state_request(active)


class PhyxioService:
    """MAYA's speech lifecycle and the public Phyxio API used by the runner/tools."""

    def __init__(self, session: SessionConfig):
        self.client_connected = False
        self.asr_queue: queue.Queue[str] = queue.Queue()

        self.session = session
        self.preferences = session.preferences
        self.voice = VoiceConfig(mic_device=_mic_device_from_env())
        self.cooldown_s = float(os.getenv("ASR_TTS_COOLDOWN", "1.2"))
        self._cooldown_until = 0.0

        self._lock = Lock()
        self._recording = False
        self._cancel_recording = Event()
        self._record_thread: Optional[Thread] = None

        self._tts_running = Event()
        self._tts_queue: queue.Queue[str] = queue.Queue()
        self._requested_agent_state = "idle"

        self._asr_client, self._asr_model = init_asr()
        self._tts = init_tts(self.preferences)
        if self._asr_client is None or self._tts is None:
            raise RuntimeError("Phyxio requires Whisper and Azure Speech credentials.")

        _list_audio_devices_once()
        self._phyxio = MyPhyxioConnector(self)
        Thread(target=self._tts_loop, daemon=True).start()

        log.info("✅ Phyxio connector ready (Whisper ASR + Azure TTS)")

    def _handle_interaction_state_request(self, active: bool):
        """
        Must return True/(True, "") to accept, or (False, reason) to deny.
        """
        log.info("[PHYXIO] change_interaction_state_request(active=%s)", active)

        with self._lock:
            if not active:
                # Only the worker clears _recording after releasing the microphone.
                self._cancel_recording.set()
                self._ui_off()
                return True, ""
            if not self.client_connected:
                return False, "I am not ready yet"
            if time.monotonic() < self._cooldown_until or self._requested_agent_state == "thinking":
                return False, "One moment…"
            if self._tts_running.is_set() or not self._tts_queue.empty():
                return False, "I'm speaking. Tap again when I finish."
            if self._recording:
                return False, "Already listening"

            self._cancel_recording.clear()
            self._recording = True
            self._start_interaction()
            self._record_thread = Thread(target=self._record_transcribe_once, daemon=True)
            try:
                self._record_thread.start()
            except RuntimeError:
                self._recording = False
                self._end_interaction()
                log.exception("Could not start the recording worker")
                return False, self.voice.error_text
            return True, ""

    # -----------------------------
    # Recording + Whisper ASR
    # -----------------------------
    def _record_transcribe_once(self) -> None:
        delivered = False
        try:
            wav, stats = self._record_until_pause()
            log.info("[mic] %s", stats)

            if self._cancel_recording.is_set():
                return

            if stats["peak"] < 0.005 or not stats["speech_seen"]:
                self._user_set_text("No speech detected. Tap and try again.")
                return

            self._ui_off()
            self._user_set_text("Transcribing…")
            text = self._transcribe_wav(wav)
            log.info("[ASR] %r", text)

            if self._cancel_recording.is_set():
                return

            if not text:
                self._user_set_text("No speech detected. Tap and try again.")
                return

            # show transcript then push to agent
            self._user_set_text(text)
            self.asr_queue.put(text)
            delivered = True

        except Exception:
            log.exception("ASR failed")
            self._user_set_text(self.voice.error_text)
        finally:
            self._ui_off()
            if not delivered and not self._tts_running.is_set() and self._tts_queue.empty():
                self._end_interaction()
            with self._lock:
                self._recording = False

    def _record_until_pause(self) -> tuple[BytesIO, dict]:
        v = self.voice
        audio_q: queue.Queue[np.ndarray] = queue.Queue()
        blocks: list[np.ndarray] = []

        silence_target = int(v.silence_seconds * v.rate)
        max_frames = int(v.max_seconds * v.rate)
        min_frames = int(v.min_seconds * v.rate)

        silence_run = 0
        total_frames = 0
        noise_floor = v.silence_threshold
        max_level = 0.0
        speech_seen = False
        last_eff = v.silence_threshold

        def cb(indata: np.ndarray, _frames: int, _time, status) -> None:
            if status:
                print(f"[audio] {status}", file=sys.stderr)
            if self._cancel_recording.is_set() or self._tts_running.is_set():
                raise sd.CallbackStop()
            audio_q.put(indata.copy())

        deadline = time.monotonic() + v.max_seconds
        with sd.InputStream(
            samplerate=v.rate,
            channels=1,
            dtype="float32",
            device=v.mic_device,
            callback=cb,
        ) as stream:
            self._asr_set_state("listening")
            self._user_set_text("Listening…")
            while True:
                if self._cancel_recording.is_set() or self._tts_running.is_set():
                    break

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    block = audio_q.get(timeout=min(v.poll_seconds, remaining))
                except queue.Empty:
                    if self._cancel_recording.is_set() or self._tts_running.is_set():
                        break
                    if not stream.active or time.monotonic() >= deadline:
                        raise OSError("Microphone stopped delivering audio.")
                    continue
                blocks.append(block)
                total_frames += len(block)

                level = float(np.sqrt(np.mean(np.square(block))) + 1e-12)
                max_level = max(max_level, level)

                if level < v.silence_threshold * 4:
                    noise_floor = 0.9 * noise_floor + 0.1 * level

                eff = max(v.silence_threshold, noise_floor * 3.0, max_level * 0.05)
                last_eff = eff

                if level >= eff:
                    speech_seen = True
                    silence_run = 0
                else:
                    silence_run = silence_run + len(block) if speech_seen else 0

                if total_frames >= max_frames:
                    break
                if speech_seen and total_frames >= min_frames and silence_run >= silence_target:
                    break

        frames = np.concatenate(blocks, axis=0) if blocks else np.zeros((0, 1), dtype=np.float32)
        peak = float(np.max(np.abs(frames))) if frames.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(frames))) + 1e-12) if frames.size else 0.0

        buf = BytesIO()
        with sf.SoundFile(buf, mode="w", samplerate=v.rate, channels=1, format="WAV") as f:
            f.write(frames)
        buf.seek(0)

        stats = {
            "seconds_captured": total_frames / v.rate,
            "peak": peak,
            "rms": rms,
            "speech_seen": speech_seen,
            "noise_floor": float(noise_floor),
            "max_level": float(max_level),
            "last_effective_threshold": float(last_eff),
            "device": v.mic_device,
        }
        return buf, stats

    def _transcribe_wav(self, wav: BytesIO) -> str:
        return transcribe(self._asr_client, self._asr_model, wav, self.preferences)

    # -----------------------------
    # Azure TTS (worker thread)
    # -----------------------------
    def speak_text(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self._tts_queue.put(text)

    def _tts_loop(self) -> None:
        while True:
            try:
                text = self._tts_queue.get(timeout=1)
            except queue.Empty:
                continue

            text = (text or "").strip()
            if not text:
                continue

            self._cancel_recording.set()
            with self._lock:
                self._tts_running.set()
            try:
                self._asr_set_state("stop")
                with self._lock:
                    self._agent_set_state("talking")

                res = self._tts.speak_text_async(text).get()
                audio = getattr(res, "audio_data", None)
                if not audio:
                    log.error("TTS returned no audio_data")
                    continue

                data, rate = sf.read(BytesIO(audio), dtype="float32")
                sd.play(data, rate)
                sd.wait()
            except Exception:
                log.exception("TTS failed")
            finally:
                with self._lock:
                    self._tts_running.clear()
                    self._cooldown_until = time.monotonic() + self.cooldown_s
                    self._agent_set_state(self._requested_agent_state)

    # -----------------------------
    # UI helpers
    # -----------------------------
    def set_agent_state(self, state: str) -> None:
        """Keep speech visible, then restore the current processing state."""
        with self._lock:
            self._requested_agent_state = state
            if not self._tts_running.is_set():
                self._agent_set_state(state)

    @staticmethod
    def _call_ui(obj: object, method_name: str, *args) -> bool:
        method = getattr(obj, method_name, None)
        if not callable(method):
            return False
        try:
            method(*args)
            return True
        except Exception as err:
            log.warning("UI call %s failed: %s", method_name, err)
            return False

    def _start_interaction(self) -> None:
        agent = self._phyxio.agent
        if agent is not None:
            self._call_ui(agent, "start_interaction")

    def _end_interaction(self) -> None:
        agent = self._phyxio.agent
        if agent is not None:
            self._call_ui(agent, "end_interaction")

    def start_interaction(self) -> None:
        self._start_interaction()

    def end_interaction(self, delay_seconds: float = 0.0) -> None:
        delay = max(float(delay_seconds or 0.0), 0.0)
        if delay <= 0:
            self._end_interaction()
            return

        def delayed_end() -> None:
            wait_started = time.monotonic()
            while self._tts_running.is_set() or not self._tts_queue.empty():
                if time.monotonic() - wait_started > 120:
                    break
                time.sleep(0.1)
            time.sleep(delay)
            self._end_interaction()

        Thread(target=delayed_end, daemon=True).start()

    def _asr_set_state(self, state: str) -> None:
        self._call_ui(self._phyxio.asr, "set_state", state)

    def _user_set_text(self, text: str) -> None:
        self._call_ui(self._phyxio.asr, "set_text", text)

    def _agent_set_text(self, text: str) -> None:
        self._call_ui(self._phyxio.agent, "set_text", text)

    def _agent_set_state(self, state: str) -> None:
        self._call_ui(self._phyxio.agent, "set_state", state)

    def _ui_off(self) -> None:
        self._asr_set_state("stop")


    def stop(self) -> None:
        self._cancel_recording.set()
        self._phyxio.stop()

    def show_text(self, text: str) -> None:
        self._agent_set_text(text)

    def get_routine_list(self) -> Any:
        return self._phyxio.routine.get_list()

    def session_history_reader(self, user_id: str, timeout_ms: int):
        """One front-end connection, with a timeout scoped to read-only history calls."""
        if user_id != self.session.user_id:
            raise ValueError("Exercise history does not match the active participant session.")
        clients = list(self._phyxio._admin_servant._clients.values())
        if len(clients) != 1:
            raise RuntimeError("Session history requires exactly one connected Phyxio client.")
        return clients[0].proxy.ice_invocationTimeout(timeout_ms)

    @staticmethod
    def _routine_id_candidates(rid: str) -> list[Any]:
        text = str(rid or "").strip()
        candidates: list[Any] = []
        if text:
            try:
                candidates.append(int(text))
            except ValueError:
                pass
            if text not in candidates:
                candidates.append(text)
        return candidates

    def _perform_routine_action(self, action: str, rid: str) -> str:
        method = getattr(self._phyxio.routine, action)
        candidates = self._routine_id_candidates(rid)
        if not candidates:
            return "Could not do that: no routine id was provided"

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                method(candidate)
                return {
                    "show": "Showing routine",
                    "run": "Started routine",
                    "calibrate": "Calibrated routine",
                }[action]
            except connect.UnableToPerform as err:
                return f"Could not do that: {err.reason}"
            except (TypeError, ValueError) as err:
                last_error = err

        return f"Could not do that: {last_error}"

    def show_routine(self, rid: str) -> str:
        return self._perform_routine_action("show", rid)

    def run_routine(self, rid: str) -> str:
        return self._perform_routine_action("run", rid)

    def calibrate_routine(self, rid: str) -> str:
        return self._perform_routine_action("calibrate", rid)
