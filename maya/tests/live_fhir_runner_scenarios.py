"""Actual main loop with real models/retrieval and typed synthetic FHIR scenarios.

RAM-only memory; simulated SDK, microphone and speech output. No ICE or UCLM I/O.
Run explicitly: python -m maya.tests.live_fhir_runner_scenarios
"""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dotenv import load_dotenv


def run():
    load_dotenv()
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
    from maya.framework.conversation.personalization import SessionConfig
    from maya.framework.memory.ontology import OntologyMemoryService
    from maya.framework.supervisor import factory, run_with_vad as runner
    from maya.tests.test_fhir_biomarkers import configured_service, observation, records

    workers, locks, streams, speech, typed = [], [], [], [], []
    # main() changes the working directory to the package root.
    output = Path(__file__).resolve().parents[2] / "output/fhir_runner_live.json"
    output.parent.mkdir(exist_ok=True)
    with TemporaryDirectory() as folder:
        service = configured_service(folder)
        service.provider.synthetic = True
        service.config.patient_map = {"synthetic-fhir-runner": "synthetic-patient"}
        service.provider.client.get.return_value = records(observation())
        turns = iter([
            "Please show my latest blood pressure measurement with its date, without interpreting it.",
            "Please interpret my latest blood pressure and explain whether it is safe given the cancer treatment I received.",
            "Doxorubicin, completed six months ago.",
            "Please show my latest blood pressure measurement with its date, without interpreting it.",
            "/exit",
        ])

        def read(prompt):
            # Unlike a person, scripted input otherwise outruns background verification.
            # The separate memory suite exercises the pending/recovery guard itself.
            for worker in workers:
                with worker._condition:
                    assert worker._condition.wait_for(lambda: not worker._scheduled_jobs, timeout=120)
            text = next(turns)
            typed.append({"prompt": prompt, "text": text})
            if len(typed) == 4:
                service.provider.client.get.return_value = []
            return text

        def memory_factory(**kwargs):
            worker = OntologyMemoryService(**kwargs, register_shutdown=False)
            workers.append(worker)
            return worker

        acquire = runner.acquire_single_instance_lock
        def acquire_lock():
            lock = acquire()
            if lock:
                locks.append(lock)
            return lock

        stream = runner.stream_updates
        def observe_stream(app, payload, config, *args, **kwargs):
            result = stream(app, payload, config, *args, **kwargs)
            snapshot = app.get_state(config)
            message = snapshot.values["messages"][-1]
            streams.append({"route": snapshot.values.get("route_type"),
                            "lookups": snapshot.values.get("biomarker_results", []),
                            "interrupts": [i.value for i in snapshot.interrupts],
                            "answer": message.content, "metadata": getattr(message, "additional_kwargs", {}),
                            "sdk_calls": service.provider.client.get.call_count})
            output.write_text(json.dumps({"streams": streams, "complete": False}, indent=2, default=str))
            return result

        try:
            with patch.dict(os.environ, {"MAYA_ENABLE_PHYXIO_IO": "0", "MAYA_RUN_THREAD_ID": "",
                                         "MAYA_RUN_LOCK": str(Path(folder) / "runner.lock")}), \
                    patch.object(factory, "get_postgres_memory", return_value=(InMemoryStore(), InMemorySaver())), \
                    patch.object(factory, "OntologyMemoryService", side_effect=memory_factory), \
                    patch.object(factory, "load_biomarker_service", return_value=service), \
                    patch.object(runner, "acquire_single_instance_lock", side_effect=acquire_lock), \
                    patch.object(runner, "init_asr", return_value=(None, None)), \
                    patch.object(runner, "init_tts", return_value=None), \
                    patch.object(runner, "speak", side_effect=lambda tts, text: speech.append(text)), \
                    patch.object(runner, "stream_updates", side_effect=observe_stream), \
                    patch("builtins.input", side_effect=read):
                runner.main(SessionConfig(user_id="synthetic-fhir-runner"))
        finally:
            for worker in workers:
                worker.close()
            for lock in locks:
                lock.close()
        checks = {
            "main_returned_and_consumed_exit": len(typed) == 5,
            "readout": bool(streams) and streams[0]["route"] == "biomarkers" and "134" in streams[0]["answer"],
            "treatment_clarification": len(streams) == 4 and "treatment" in str(streams[1]["interrupts"]).lower(),
            "resume_without_refetch": len(streams) == 4 and streams[1]["sdk_calls"] == streams[2]["sdk_calls"],
            "resumed_with_evidence": len(streams) == 4 and bool(streams[2]["metadata"].get("presentation", {}).get("references")),
            "ambiguous_empty_unavailable": bool(streams) and bool(streams[-1]["lookups"])
                and streams[-1]["lookups"][0]["status"] == "unavailable",
            "speech_boundary_exercised": any("134" in text for text in speech),
            "only_synthetic_patient_queried": all(call.kwargs == {"subject": "synthetic-patient"}
                                                 for call in service.provider.client.get.call_args_list),
            "locks_closed": all(lock.closed for lock in locks),
        }
    report = {"passed": all(checks.values()), "checks": checks, "streams": streams, "typed": typed,
              "speech": speech, "scope": "Actual runner, models and retrieval; synthetic SDK, RAM memory and simulated audio"}
    output.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"passed": report["passed"], "checks": checks}), flush=True)
    assert report["passed"], f"Review {output}"
    return report


if __name__ == "__main__":
    run()
