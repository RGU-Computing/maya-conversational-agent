"""Live models/retrieval through run_with_vad; capture speech instead of playing it.

Run explicitly: python -m maya.tests.live_health_presentation_scenarios
Participant memory is isolated in RAM. ICE and audio hardware are not exercised.
"""

import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

from dotenv import load_dotenv


def run():
    load_dotenv()
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
    from maya.framework.conversation.personalization import SessionConfig
    from maya.framework.conversation.responses import load_response_policy
    from maya.framework.supervisor import factory, run_with_vad as runner

    results = []
    policy = load_response_policy()
    store = InMemoryStore()
    with patch.object(factory, "get_postgres_memory", return_value=(store, InMemorySaver())), \
            patch.object(factory.OntologyMemoryService, "schedule", return_value=False):
        graph = factory.build_supervisor()

    cases = [
        ("definition", "What is cardiotoxicity after cancer treatment?", "definition"),
        ("worried", "I'm worried about heart problems after cancer treatment. Can you explain, in general terms, why follow-up heart monitoring is used?", "worried"),
        ("cached_definition", "What is cardiotoxicity after cancer treatment?", "definition"),
    ]
    for name, question, participant in cases:
        print(f"Running {name} through run_with_vad...", flush=True)
        session = SessionConfig(user_id=f"live-health-presentation-{participant}")
        inputs = iter([question, None])
        shown, spoken, configs, prompts = [], [], [], []
        stream = runner.stream_updates

        def capture_stream(app, payload, config, phyxio_io=None, **kwargs):
            configs.append(config)
            return stream(app, payload, config, phyxio_io, **kwargs)

        def read(prompt, *args, **kwargs):
            if prompt:
                prompts.append(prompt)
                raise AssertionError("Unexpected clarification in a general education scenario")
            value = next(inputs)
            if value is not None:
                assert not spoken, "MAYA spoke before receiving the user's question"
            return value

        with patch.object(runner.os, "chdir"), patch.object(runner.atexit, "register"), \
                patch.object(runner, "acquire_single_instance_lock", return_value=Mock()), \
                patch.object(runner, "build_supervisor", return_value=graph), \
                patch.object(runner, "init_phyxio_io", return_value=None), \
                patch.object(runner, "init_asr", return_value=(None, None)), \
                patch.object(runner, "init_tts", return_value=Mock()), \
                patch.object(runner, "read_text", side_effect=read), \
                patch.object(runner, "stream_updates", side_effect=capture_stream), \
                patch.object(runner, "speak", side_effect=lambda tts, text: spoken.append(text)), \
                patch("builtins.print", side_effect=lambda *args, **kwargs: shown.append(" ".join(map(str, args)))):
            runner.main(session)

        message = graph.get_state(configs[-1]).values["messages"][-1]
        presentation = message.additional_kwargs["presentation"]
        trace = message.additional_kwargs["workflow_trace"][0]
        expected_speech = [policy.health_processing_status, presentation["speech_text"]]
        assert spoken == expected_speech
        assert all(policy.health_processing_status not in str(m.content)
                   for m in graph.get_state(configs[-1]).values["messages"])
        assert spoken[-1] == presentation["speech_text"]
        assert any(presentation["display_text"] in text for text in shown)
        assert presentation["display_text"].endswith(policy.health_disclaimer)
        assert presentation["speech_text"].endswith(policy.health_disclaimer)
        assert presentation["references"]
        assert all(ref["file_name"] not in spoken[-1] for ref in presentation["references"] if ref["file_name"])
        assert trace["memory_lookup"] == ("exact" if name == "cached_definition" else "miss")
        results.append({"scenario": name, "question": question, "presentation": presentation,
                        "retrieved_excerpt_count": len(trace["citations"]),
                        "memory_lookup": trace["memory_lookup"], "spoken_messages": spoken,
                        "delivery_checks_passed": True})

    output = Path(__file__).resolve().parents[2] / "output" / "health_presentation_live_results.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({"scope": "Live Azure and literature retrieval through main runner; captured TTS text; no ICE/audio hardware", "results": results}, indent=2))
    print(f"Passed {len(results)} live delivery scenarios. Saved {output}")
    return results


if __name__ == "__main__":
    run()
