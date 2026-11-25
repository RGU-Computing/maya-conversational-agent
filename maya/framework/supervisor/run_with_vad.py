from __future__ import annotations

import argparse
import os
import sys
import queue
import select
import atexit
import fcntl
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from math import isfinite
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
from dotenv import load_dotenv
from langsmith.utils import tracing_is_enabled
from openai import APIError, BadRequestError
import azure.cognitiveservices.speech as speechsdk
from langgraph.types import Command

ROOT = Path(__file__).resolve().parents[2]

if __package__ in {None, ""}:
    sys.path.append(str(ROOT.parent))

from maya.framework.conversation.personalization import Preferences, SessionConfig, load_session
from maya.framework.integrations.speech import init_asr, init_tts, transcribe
from maya.framework.conversation.responses import ResponseText, response_policy, response_text

load_dotenv()


@dataclass(frozen=True)
class VoiceConfig:
    seconds: float
    rate: int
    poll_seconds: float = 0.1

    def __post_init__(self):
        if any(not isfinite(value) or value <= 0 for value in (self.seconds, self.poll_seconds)):
            raise ValueError("Voice duration and polling interval must be finite and positive.")


VAD_AGGRESSIVENESS = int(os.getenv("MAYA_VAD_AGGRESSIVENESS", "3"))
VAD_FRAME_MS = int(os.getenv("MAYA_VAD_FRAME_MS", "10"))
VAD_PADDING_MS = int(os.getenv("MAYA_VAD_PADDING_MS", "100"))
VAD_SILENCE_MS = int(os.getenv("MAYA_VAD_SILENCE_MS", "200"))
VAD_RMS_THRESHOLD = float(os.getenv("MAYA_VAD_RMS_THRESHOLD", "0.006"))
VOICE_SECONDS = float(os.getenv("MAYA_VOICE_SECONDS", "20"))
VOICE_RATE = int(os.getenv("MAYA_VOICE_SAMPLE_RATE", "16000"))
PHYXIO_END_INTERACTION_DELAY = float(os.getenv("MAYA_PHYXIO_END_INTERACTION_DELAY", "4.5"))


def speak(tts: speechsdk.SpeechSynthesizer | None, text: str) -> None:
    if not tts or not text:
        return
    try:
        result = tts.speak_text_async(text).get()
        audio_bytes = getattr(result, "audio_data", None)
        if audio_bytes:
            buffer = BytesIO(audio_bytes)
            data, rate = sf.read(buffer, dtype="float32")
            sd.play(data, rate)
            sd.wait()
    except Exception as tts_err:
        print(f"[tts] {type(tts_err).__name__}")


def init_phyxio_io(session: SessionConfig):
    if os.getenv("MAYA_ENABLE_PHYXIO_IO", "1").strip().lower() not in {"1", "true", "yes"}:
        return None
    try:
        from maya.framework.integrations.phyxio.bridge import get_phyxio_service

        return get_phyxio_service(session)
    except ValueError:
        raise
    except Exception as phyxio_err:
        print(f"[phyxio] disabled: {type(phyxio_err).__name__}")
        return None


def speak_phyxio(phyxio_io, text: str, *, speech_text: str | None = None) -> None:
    if not phyxio_io or not text:
        return
    try:
        phyxio_io.show_text(text)
        phyxio_io.speak_text(text if speech_text is None else speech_text)
    except Exception as phyxio_err:
        print(f"[phyxio][tts] {type(phyxio_err).__name__}")


def set_phyxio_state(phyxio_io, state: str) -> None:
    if not phyxio_io:
        return
    try:
        set_state = getattr(phyxio_io, "set_agent_state", None)
        if callable(set_state):
            set_state(state)
    except Exception as phyxio_err:
        print(f"[phyxio][state] {type(phyxio_err).__name__}")


def end_phyxio_interaction(phyxio_io, delay_seconds: float = PHYXIO_END_INTERACTION_DELAY) -> None:
    if not phyxio_io:
        return
    try:
        end_interaction = getattr(phyxio_io, "end_interaction", None)
        if callable(end_interaction):
            end_interaction(delay_seconds)
    except Exception as phyxio_err:
        print(f"[phyxio][end_interaction] {type(phyxio_err).__name__}")


def speak_output(
    tts: speechsdk.SpeechSynthesizer | None,
    phyxio_io,
    text: str,
    *,
    speech_text: str | None = None,
    end_interaction: bool = False,
) -> None:
    if phyxio_io is not None:
        speak_phyxio(phyxio_io, text, speech_text=speech_text)
        if end_interaction:
            end_phyxio_interaction(phyxio_io)
    else:
        speak(tts, text if speech_text is None else speech_text)


def _interactive_output() -> bool:
    isatty = getattr(sys.stdout, "isatty", None)
    return bool(isatty and isatty())


def read_text(
    prompt_text: str | None,
    tts: speechsdk.SpeechSynthesizer | None,
    asr_client: object | None,
    asr_model: str | None,
    voice: VoiceConfig,
    phyxio_io=None,
    *,
    allow_exit: bool = False,
    blank_returns_none: bool = False,
    preferences: Preferences = Preferences(),
) -> str | None:
    if prompt_text:
        speak_output(tts, phyxio_io, prompt_text)
        input_prompt = f"MAYA: {prompt_text}\nYou: "
    else:
        input_prompt = "You: "

    def capture_voice() -> str:
        print(f"Recording... speak and pause to stop (max {voice.seconds:.1f}s)")

        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        frame_ms = VAD_FRAME_MS
        padding_ms = VAD_PADDING_MS
        silence_ms = VAD_SILENCE_MS
        rms_threshold = VAD_RMS_THRESHOLD

        frame_samples = int(voice.rate * frame_ms / 1000)
        max_frames = int(voice.seconds * voice.rate)
        padding_frames = max(1, int(padding_ms / frame_ms))
        silence_frames = max(1, int(silence_ms / frame_ms))

        audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        recorded_blocks: list[np.ndarray] = []
        ring = deque(maxlen=padding_frames)
        silence_run = 0
        total_frames = 0
        triggered = False

        def callback(indata: np.ndarray, _frames: int, _time, status) -> None:
            if status:
                print(f"[audio] {status}", file=sys.stderr)
            audio_queue.put(indata.copy())

        mic_device = os.getenv("MAYA_MIC_DEVICE")
        deadline = time.monotonic() + voice.seconds
        with sd.InputStream(
            samplerate=voice.rate,
            channels=1,
            dtype="int16",
            device=int(mic_device) if mic_device else None,
            blocksize=frame_samples,
            callback=callback,
        ) as stream:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    block = audio_queue.get(timeout=min(voice.poll_seconds, remaining)).reshape(-1)
                except queue.Empty:
                    if not stream.active:
                        raise OSError("Microphone stopped delivering audio.")
                    continue
                total_frames += len(block)
                pcm = block.tobytes()
                rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)) / 32768.0)
                is_speech = rms > rms_threshold and vad.is_speech(pcm, voice.rate)

                if not triggered:
                    ring.append(block)
                    if is_speech:
                        triggered = True
                        recorded_blocks.extend(ring)
                        ring.clear()
                    if total_frames >= max_frames:
                        break
                    continue

                recorded_blocks.append(block)
                if is_speech:
                    silence_run = 0
                else:
                    silence_run += 1

                if silence_run >= silence_frames or total_frames >= max_frames:
                    break

        if not recorded_blocks:
            return ""

        frames = np.concatenate(recorded_blocks, axis=0)
        buffer = BytesIO()
        with sf.SoundFile(
            buffer, mode="w", samplerate=voice.rate, channels=1, format="WAV"
        ) as wav_file:
            wav_file.write(frames)
        buffer.seek(0)

        return transcribe(asr_client, asr_model, buffer, preferences)

    while True:
        if phyxio_io is not None:
            if _interactive_output():
                print(input_prompt, end="", flush=True)
            while True:
                try:
                    mirror_text = phyxio_io.asr_queue.get_nowait()
                except queue.Empty:
                    mirror_text = ""
                except Exception:
                    mirror_text = ""
                if isinstance(mirror_text, str) and mirror_text.strip():
                    spoken = mirror_text.strip()
                    print()
                    return spoken
                
                if not sys.stdin.isatty():
                    time.sleep(0.1)
                    continue
                try:
                    ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                except (ValueError, OSError):
                    ready = []
                if ready:
                    raw = sys.stdin.readline()
                    if raw == "":
                        print()
                        return None
                    user_input = raw.strip()
                    break
        else:
            try:
                user_input = input(input_prompt if _interactive_output() else "").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return None

        if allow_exit and user_input in {"/exit", "/quit"}:
            return None

        wants_voice = user_input == "/voice" or (user_input == "" and asr_client and phyxio_io is None)
        if wants_voice:
            if not asr_client or not asr_model:
                if user_input == "/voice":
                    print("Whisper key not configured. Please provide text input instead.")
                    continue
                return None if blank_returns_none else ""
            try:
                spoken_text = capture_voice()
            except KeyboardInterrupt:
                print("\nVoice input cancelled.")
                continue
            except Exception as voice_err:
                print(f"[voice] {type(voice_err).__name__}")
                print(os.getenv("MAYA_VOICE_ERROR_TEXT", "Voice input is temporarily unavailable. Please tap to try again."))
                continue
            if not spoken_text:
                print("No speech detected. Try again.")
                continue
            return spoken_text

        if user_input:
            return user_input

        if phyxio_io is not None:
            continue

        if blank_returns_none:
            return None
        return ""


def stream_updates(app, payload, base_config, phyxio_io=None, *, on_health_processing=None):
    final_text = None
    interrupts = []
    seen_health_rag_traces: set[tuple[bool, str, bool, bool]] = set()

    set_phyxio_state(phyxio_io, "thinking")
    try:
        for _, mode, update in app.stream(
            payload,
            config=base_config,
            stream_mode=["updates", "custom"],
            subgraphs=True,
        ):
            if not isinstance(update, dict):
                continue
            if mode == "custom":
                if update.get("event") == "health_processing_started" and on_health_processing:
                    on_health_processing()
                continue

            if "__interrupt__" in update:
                interrupts.extend(update.get("__interrupt__", []))
                continue

            for node, node_payload in update.items():
                if node == "__interrupt__" or node_payload is None:
                    continue

                if isinstance(node_payload, dict):
                    messages = node_payload.get("messages", [])
                elif isinstance(node_payload, list):
                    messages = node_payload
                else:
                    messages = getattr(node_payload, "messages", [])

                if messages is None:
                    continue
                if isinstance(messages, dict):
                    messages = list(messages.values())
                elif not isinstance(messages, list):
                    messages = [messages]
                if not messages:
                    continue
                last = messages[-1]
                if isinstance(last, dict):
                    role = last.get("role") or last.get("type")
                    name = last.get("name")
                    content = last.get("content")
                    additional_kwargs = last.get("additional_kwargs") or {}
                else:
                    role = getattr(last, "type", None)
                    name = getattr(last, "name", None)
                    content = getattr(last, "content", None)
                    additional_kwargs = getattr(last, "additional_kwargs", {}) or {}

                if name:
                    print(f"[{node}] {role} :: {name}")

                if clarification := additional_kwargs.get("clarification"):
                    print(
                        "[health_rag] clarification :: "
                        f"status={clarification['status']} "
                        f"question_type={clarification.get('question_type', 'unknown')} "
                        f"needs_clarification={clarification.get('needs_clarification', 'unknown')} "
                        f"missing_fields={clarification.get('material_missing_fields', [])} "
                        f"error_type={clarification.get('error_type', 'none')}"
                    )

                workflow_trace = additional_kwargs.get("workflow_trace")
                if isinstance(workflow_trace, list):
                    for item in workflow_trace:
                        if not isinstance(item, dict):
                            continue
                        stage = str(item.get("stage", "")).strip()
                        if stage != "health_rag":
                            continue
                        memory_hit = bool(item.get("memory_hit"))
                        memory_lookup = str(item.get("memory_lookup", ""))
                        retrieval_used = bool(item.get("retrieval_used"))
                        summary_stored = bool(item.get("summary_stored"))
                        trace_key = (memory_hit, memory_lookup, retrieval_used, summary_stored)
                        if trace_key in seen_health_rag_traces:
                            continue
                        seen_health_rag_traces.add(trace_key)
                        print(
                            "[health_rag] trace :: "
                            f"memory_hit={memory_hit} "
                            f"memory_lookup={memory_lookup or 'unknown'} "
                            f"retrieval_used={retrieval_used} "
                            f"summary_stored={summary_stored}"
                        )

                if role in {"assistant", "ai"} and content:
                    final_text = response_text(content, additional_kwargs)
    finally:
        set_phyxio_state(phyxio_io, "idle")

    return final_text, interrupts


def is_tool_call_history_error(err: Exception) -> bool:
    if not isinstance(err, BadRequestError):
        return False
    text = str(err)
    return (
        "assistant message with 'tool_calls' must be followed by tool messages" in text
        and "tool_call_id" in text
    )


def acquire_single_instance_lock():
    lock_path = os.getenv("MAYA_RUN_LOCK", "/tmp/maya_run_with_vad.lock")
    lock_file = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        owner = "unknown"
        try:
            with open(lock_path, "r", encoding="utf-8") as existing:
                owner = (existing.read() or "").strip() or owner
        except Exception:
            pass
        print(
            f"[runner] Another run_with_vad instance is active (pid={owner}). "
            "Stop it before starting a new one."
        )
        lock_file.close()
        return None

    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    return lock_file


def build_supervisor():
    # Keep the existing entry point for the latency runner, without model setup on import.
    from maya.framework.supervisor.factory import build_supervisor as build

    return build()


def main(session: SessionConfig | None = None) -> None:
    if tracing_is_enabled():
        raise RuntimeError("Disable LangSmith tracing before starting a participant session.")
    session = session or load_session()
    os.chdir(ROOT)
    lock_file = acquire_single_instance_lock()
    if lock_file is None:
        return
    atexit.register(lock_file.close)

    # Prepare notices before accepting input; request errors must not need an LLM.
    policy = response_policy(session.preferences.response_language)

    base_config = session.graph_config(thread_id=os.getenv("MAYA_RUN_THREAD_ID"))

    app = build_supervisor()

    phyxio_io = init_phyxio_io(session)
    if phyxio_io is not None:
        atexit.register(phyxio_io.stop)
    asr_client, asr_model = (None, None) if phyxio_io is not None else init_asr()
    voice = VoiceConfig(
        seconds=VOICE_SECONDS,
        rate=VOICE_RATE,
        poll_seconds=float(os.getenv("MAYA_VOICE_POLL_SECONDS", "0.1")),
    )
    tts = None if phyxio_io is not None else init_tts(session.preferences)

    if phyxio_io is not None:
        print("Type a message or speak from the Phyxio mirror. (Ctrl+C to exit.)")
    elif asr_client:
        print("Type a message or press Enter for voice input. (Ctrl+C to exit.)")
    else:
        print("Type your message (blank line to exit).")

    while True:
        user_message = read_text(
            None,
            tts,
            asr_client,
            asr_model,
            voice,
            phyxio_io,
            allow_exit=True,
            blank_returns_none=True,
            preferences=session.preferences,
        )
        if user_message is None:
            break

        payload = {"messages": [{"role": "user", "content": user_message}]}

        # One acknowledgement per user turn, including graph retries and resumes.
        with ThreadPoolExecutor(max_workers=1) as speech_worker:
            progress_announced = False

            def delayed_acknowledgement():
                nonlocal progress_announced
                if response_ready.wait(policy.health_processing_delay_seconds):
                    return
                progress_announced = True
                text = policy.health_processing_status
                print(f"MAYA: {text}")
                speak_output(tts, phyxio_io, text)

            def announce_health_processing():
                nonlocal progress_speech
                if not progress_announced and progress_speech is None:
                    progress_speech = speech_worker.submit(delayed_acknowledgement)

            while True:
                response_ready = Event()
                progress_speech = None
                try:
                    try:
                        final_text, interrupts = stream_updates(
                            app, payload, base_config, phyxio_io, on_health_processing=announce_health_processing,
                        )
                    except Exception as err:
                        if not is_tool_call_history_error(err):
                            raise
                        base_config = session.graph_config()
                        print("[runner] recovered from invalid stored tool-call history; retrying in a new thread.")
                        final_text, interrupts = stream_updates(
                            app, payload, base_config, phyxio_io, on_health_processing=announce_health_processing,
                        )
                except APIError as err:
                    # Do not automatically replay a rejected/failed message. The next
                    # user request starts clean, without an unfinished tool/interrupt turn.
                    base_config = session.graph_config()
                    text = policy.request_unavailable_text
                    final_text, interrupts = ResponseText(text, text), []
                    print(f"[runner] model request failed ({type(err).__name__}); ready for a new request.")
                finally:
                    # Cancel a pending acknowledgement; finish one already started.
                    response_ready.set()
                    if progress_speech is not None:
                        progress_speech.result()
                if interrupts:
                    answers: list[str] = []
                    for interrupt_ in interrupts:
                        prompt = str(interrupt_.value)
                        answer = read_text(
                            prompt, tts, asr_client, asr_model, voice, phyxio_io,
                            preferences=session.preferences,
                        )
                        answers.append(answer)
                    resume_value = answers[0] if len(answers) == 1 else answers
                    payload = Command(resume=resume_value)
                    continue
                break

        if final_text:
            if _interactive_output():
                print(f"\n***************\n\nMAYA: {final_text.display_text}")
            speak_output(tts, phyxio_io, final_text.display_text,
                         speech_text=final_text.speech_text, end_interaction=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MAYA with optional onboarding settings.")
    parser.add_argument("--session-config", type=Path, help="JSON participant ID and preferences")
    args = parser.parse_args()
    try:
        main(load_session(args.session_config))
    except KeyboardInterrupt:
        pass
