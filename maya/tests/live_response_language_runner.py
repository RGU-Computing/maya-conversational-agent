"""Actual main loop, cloud ASR/TTS and models; generated microphone input, no playback.

Run explicitly: python -m maya.tests.live_response_language_runner
Session JSON uses the mirror's speech choices; RAM memory and independent lock.
No native connector or participant storage is used. Physical audio is a separate check.
"""
import json
import os
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dotenv import load_dotenv


UTTERANCES = {
    'en': 'Maya tell me one interesting fact about the Moon.',
    'ro': 'Maya spune-mi un fapt interesant despre Lună.',
    'it': 'Maya raccontami una curiosità sulla Luna.',
    'es': 'Maya dime un dato interesante sobre la Luna.',
    'fi': 'Maya kerro minulle yksi kiinnostava asia Kuusta.',
    'pt-PT': 'Maya conta-me uma curiosidade sobre a Lua.',
    'fr': 'Maya raconte-moi un fait intéressant sur la Lune.',
}

HEALTH_QUESTIONS = {
    'en': 'In general, what is cardiotoxicity after cancer treatment?',
    'ro': 'În general, ce este cardiotoxicitatea după tratamentul cancerului?',
    'it': 'In generale, che cosa è la cardiotossicità dopo il trattamento del cancro?',
    'es': 'En general, ¿qué es la cardiotoxicidad después del tratamiento del cáncer?',
    'fi': 'Mitä syöpähoidon jälkeinen kardiotoksisuus tarkoittaa yleisesti?',
    'pt-PT': 'Em geral, o que é a cardiotoxicidade após o tratamento do cancro?',
    'fr': 'En général, qu’est-ce que la cardiotoxicité après un traitement contre le cancer ?',
}


def run(languages=(), *, utterance=None, output_path=None, speech_preferences=None):
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / '.env')
    os.environ.update(LANGSMITH_TRACING='false', LANGCHAIN_TRACING_V2='false', MAYA_ENABLE_PHYXIO_IO='0')
    import numpy as np
    import soundfile as sf
    import azure.cognitiveservices.speech as speechsdk
    from langgraph.store.memory import InMemoryStore
    from langgraph.checkpoint.memory import InMemorySaver
    from maya.framework.conversation.personalization import load_session, RESPONSE_LANGUAGES
    from maya.framework.memory.ontology import OntologyMemoryService
    from maya.framework.supervisor import factory, run_with_vad as runner
    suffix = '_' + '_'.join(languages) if languages else ''
    if utterance:
        if len(languages) != 1:
            raise ValueError('A custom utterance requires exactly one language.')
        suffix += '_custom'
    output = Path(output_path) if output_path is not None else root / f'output/response_language_runner{suffix}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    if speech_preferences is None:
        base = Path((root / 'output/validation_location.txt').read_text().strip())
        speech_preferences = load_session(base / 'session.json').preferences.model_dump()
    report = {'scope': 'Actual runner, VAD, Azure Whisper, models/retrieval and Azure TTS. Generated audio in simulated microphone callbacks; playback intercepted. RAM memory.', 'cases': []}
    workers, locks = [], []

    def memory_factory(**kwargs):
        worker = OntologyMemoryService(**kwargs, register_shutdown=False)
        workers.append(worker)
        return worker

    try:
        with TemporaryDirectory() as folder, \
                patch.object(factory, 'get_postgres_memory', return_value=(InMemoryStore(), InMemorySaver())), \
                patch.object(factory, 'OntologyMemoryService', side_effect=memory_factory):
            graph = factory.build_supervisor()
            for language in languages or RESPONSE_LANGUAGES.languages:
                directory = Path(folder)
                path = directory / 'session.json'
                path.write_text(json.dumps({'user_id': f'language-runner-{language}',
                    'preferences': {**speech_preferences, 'response_language': language}}))
                session = load_session(path)
                config = speechsdk.SpeechConfig(subscription=os.environ['AZURE_SPEECH_KEY'], region=os.environ['AZURE_SPEECH_REGION'])
                config.speech_synthesis_voice_name = session.preferences.tts_voice
                config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm)
                spoken_input = utterance or UTTERANCES[language]
                source = speechsdk.SpeechSynthesizer(speech_config=config, audio_config=None).speak_text_async(spoken_input).get()
                assert source.reason == speechsdk.ResultReason.SynthesizingAudioCompleted
                samples, rate = sf.read(BytesIO(source.audio_data), dtype='int16')
                assert rate == 16000
                samples = np.concatenate((samples, np.zeros(16000, dtype=np.int16)))
                captured, spoken, played, streams, inputs = [], [], [], [], []
                turns = iter(['/voice', HEALTH_QUESTIONS[language], '/exit'])
                startup_silent = []

                class Microphone:
                    active = True
                    def __init__(self, **kwargs):
                        self.kwargs = kwargs
                        streams.append(self)
                    def __enter__(self):
                        size = self.kwargs['blocksize']
                        for offset in range(0, len(samples), size):
                            frame = samples[offset:offset+size]
                            if len(frame) < size:
                                frame = np.pad(frame, (0, size-len(frame)))
                            self.kwargs['callback'](frame[:, None], size, None, None)
                        return self
                    def __exit__(self, *args):
                        self.active = False

                def read(prompt):
                    for worker in workers:
                        with worker._condition:
                            assert worker._condition.wait_for(lambda: not worker._scheduled_jobs, timeout=120)
                    if not inputs:
                        startup_silent.append(not spoken)
                    value = next(turns)
                    inputs.append(value)
                    return value

                stream = runner.stream_updates
                def observe(app, payload, graph_config, *args, **kwargs):
                    result = stream(app, payload, graph_config, *args, **kwargs)
                    state = app.get_state(graph_config)
                    message = state.values['messages'][-1]
                    captured.append({'input': payload, 'answer': message.content,
                        'metadata': message.additional_kwargs, 'route': state.values.get('route_type'),
                        'preferences': graph_config['configurable']['preferences'],
                        'interrupts': [i.value for i in state.interrupts]})
                    return result

                speak = runner.speak
                def observe_speech(tts, text):
                    spoken.append(text)
                    return speak(tts, text)
                acquire = runner.acquire_single_instance_lock
                def acquire_lock():
                    lock = acquire()
                    if lock:
                        locks.append(lock)
                    return lock

                with patch.dict(os.environ, {'MAYA_RUN_LOCK': str(directory / 'runner.lock'), 'MAYA_RUN_THREAD_ID': ''}), \
                        patch.object(runner, 'build_supervisor', return_value=graph), \
                        patch.object(runner, 'acquire_single_instance_lock', side_effect=acquire_lock), \
                        patch.object(runner.sd, 'InputStream', Microphone), \
                        patch.object(runner.sd, 'play', side_effect=lambda data, sample_rate: played.append({'samples': len(data), 'rate': sample_rate})), \
                        patch.object(runner.sd, 'wait'), \
                        patch.object(runner, 'speak', side_effect=observe_speech), \
                        patch.object(runner, 'stream_updates', side_effect=observe), \
                        patch('builtins.input', side_effect=read):
                    try:
                        runner.main(session)
                    finally:
                        for lock in locks:
                            lock.close()
                refs = captured[-1]['metadata'].get('presentation', {}).get('references', []) if captured else []
                checks = {'startup_silent': startup_silent == [True],
                    'two_turns_and_exit': len(captured) == 2 and inputs[-1] == '/exit',
                    'asr_reached_supervisor': bool(captured) and bool(captured[0]['input']['messages'][0]['content']),
                    'spoken_question_direct_route': bool(captured) and captured[0]['route'] == 'direct_answer',
                    'selected_language_propagated': all(c['preferences']['response_language'] == language for c in captured),
                    'health_retrieval': len(captured) == 2 and bool(refs) and captured[-1]['route'] == 'health_rag',
                    'reference_exclusion': bool(refs) and all(ref['file_name'] not in text for ref in refs for text in spoken if ref['file_name']),
                    'tts_audio_for_every_output': len(played) == len(spoken) and bool(played) and all(p['samples'] > 0 for p in played),
                    'final_speech_last': bool(spoken) and bool(captured) and spoken[-1] == captured[-1]['metadata'].get('presentation', {}).get('speech_text'),
                    'microphone_closed': bool(streams) and all(not s.active for s in streams),
                    'locks_closed': all(lock.closed for lock in locks)}
                report['cases'].append({'language': language, 'passed': all(checks.values()), 'checks': checks,
                    'utterance': spoken_input, 'captured': captured, 'spoken': spoken, 'audio': played})
                output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
                print(json.dumps({'language': language, 'passed': all(checks.values()), 'checks': checks}), flush=True)
    finally:
        for worker in workers:
            worker.close()
        for lock in locks:
            lock.close()
    report['passed'] = all(row['passed'] for row in report['cases'])
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('languages', nargs='*')
    parser.add_argument('--utterance', help='Additional spoken-input case for a single language; keeps a separate report.')
    args = parser.parse_args()
    raise SystemExit(0 if run(args.languages, utterance=args.utterance)['passed'] else 1)
