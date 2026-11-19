"""Exercise real LangChain middleware and capture actual provider request bodies.

All identities and measurements are synthetic; HTTP never leaves MockTransport.
Workflow tests separately exercise routing and local stores with controlled models.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
import unittest
from unittest.mock import patch

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, dynamic_prompt
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel

from maya.framework.integrations.models import model_call
from maya.framework.integrations.privacy import PrivacyMiddleware, privacy_config


class Record(BaseModel):
    text: str


class PrivacyBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}))
        self.requests = []

        def serve(request):
            body = json.loads(request.content)
            self.requests.append(body)
            text = body["messages"][-1]["content"]
            content = json.dumps({"text": text}) if body.get("response_format") else text
            answer = {"role": "assistant", "content": content}
            if body.get("tools"):
                name = body["tools"][0]["function"]["name"]
                answer = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_fixture", "type": "function", "function": {
                        "name": name, "arguments": json.dumps({"text": text}),
                    },
                }]}
            return httpx.Response(200, json={"id": "fixture", "model": "offline", "object": "chat.completion",
                "choices": [{"index": 0, "message": answer, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

        transport = httpx.MockTransport(serve)
        client = httpx.Client(transport=transport)
        self.addCleanup(client.close)
        self.model = AzureChatOpenAI(azure_endpoint="https://privacy.invalid", azure_deployment="offline",
            api_key="synthetic-key", api_version="2024-02-15-preview", max_retries=0,
            http_client=client, http_async_client=httpx.AsyncClient(transport=transport))
        self.config = privacy_config({"configurable": {"user_id": "participant-alpha"}},
                                     {"name": "Alice Fernández", "patient_id": "patient-alpha"})

    def sent(self):
        return json.dumps(self.requests[-1], ensure_ascii=False)

    def test_system_history_tools_and_typed_response_are_protected_without_state_mutation(self):
        messages = [
            SystemMessage(content='Profile: {"name":"Alice Fernández","job":null}'),
            HumanMessage(content="participant-alpha, alice@example.org", name="participant-alpha"),
            AIMessage(content="Alice Fernández", tool_calls=[{
                "name": "lookup", "id": "call_prior", "args": {"patient_id": "patient-alpha"},
            }]),
            ToolMessage(content='{"subject":{"reference":"Patient/patient-alpha"},"value":116}', tool_call_id="call_prior"),
            HumanMessage(content="Alice Fernández takes Bisoprolol 2.5 mg; BP 116/67 mmHg at 15:51."),
        ]
        before = deepcopy(messages)
        result = model_call(self.model, Record, strict=True).invoke(messages, self.config)
        for value in ("participant-alpha", "Alice Fernández", "alice@example.org", "patient-alpha"):
            self.assertNotIn(value, self.sent())
        self.assertEqual(result.text, messages[-1].content)
        self.assertEqual(messages, before)
        self.assertEqual(len(self.requests), 1)
        for value in ("Bisoprolol", "2.5 mg", "116/67 mmHg", "15:51", '"name": "lookup"'):
            self.assertIn(value, self.sent())
        self.assertTrue(self.requests[0]["response_format"]["json_schema"]["strict"])
        self.assertRegex(self.requests[0]["messages"][2]["name"], r"^[a-zA-Z0-9_-]{1,64}$")

    def test_function_calling_preserves_local_memory_text_and_makes_one_call(self):
        text = "Alice Fernández takes Doxorubicin."
        result = model_call(self.model, Record, method="function_calling").invoke([HumanMessage(content=text)], self.config)
        self.assertEqual(result.text, text)
        self.assertNotIn("Alice Fernández", self.sent())
        self.assertIn("Doxorubicin", self.sent())
        self.assertEqual(len(self.requests), 1)

    def test_provider_schema_and_settings_match_the_previous_model_api(self):
        for strict in (None, True):
            with self.subTest(strict=strict):
                self.model.with_structured_output(Record, strict=strict).invoke([HumanMessage(content="Clinical question")])
                before = deepcopy(self.requests[-1])
                model_call(self.model, Record, strict=strict).invoke([HumanMessage(content="Clinical question")])
                self.assertEqual(self.requests[-1], before)

    def test_async_prompt_chain_has_the_same_protection(self):
        chain = ChatPromptTemplate.from_messages([("human", "{text}")]) | model_call(self.model, Record)
        result = asyncio.run(chain.ainvoke({"text": "Alice Fernández, participant-alpha"}, self.config))
        self.assertEqual(result.text, "Alice Fernández, participant-alpha")
        self.assertNotIn("participant-alpha", self.sent())
        self.assertNotIn("Alice Fernández", self.sent())

    def test_guard_runs_after_dynamic_prompt_and_preserves_original_agent_state(self):
        @dynamic_prompt
        def instructions(request):
            return "Known patient: patient-alpha."
        agent = create_agent(self.model, tools=[], middleware=[instructions, PrivacyMiddleware()])
        result = agent.invoke({"messages": [HumanMessage(content="Alice Fernández")]}, self.config)
        self.assertEqual(result["messages"][-1].content, "Alice Fernández")
        self.assertNotIn("patient-alpha", self.sent())

    def test_nested_model_call_inherits_identity_from_agent_runtime(self):
        @before_model
        def rewrite(state, runtime):
            # HealthRAG's rewrite uses the inherited config rather than passing
            # identity explicitly. Both this call and the final answer need it.
            model_call(self.model, Record).invoke([HumanMessage(content="Alice Fernández, patient-alpha")])
        agent = create_agent(self.model, tools=[], middleware=[rewrite, PrivacyMiddleware()])
        agent.invoke({"messages": [HumanMessage(content="Alice Fernández, patient-alpha")]}, self.config)
        self.assertEqual(len(self.requests), 2)
        for body in self.requests:
            self.assertNotIn("Alice Fernández", json.dumps(body, ensure_ascii=False))
            self.assertNotIn("patient-alpha", json.dumps(body))

    def test_model_boundary_does_not_add_database_checkpoints(self):
        saver = InMemorySaver()
        graph = StateGraph(MessagesState)
        def answer(state):
            return {"messages": [model_call(self.model).invoke(state["messages"])]}
        graph.add_node("answer", answer)
        graph.add_edge(START, "answer")
        graph.add_edge("answer", END)
        app = graph.compile(checkpointer=saver)
        config = {**self.config, "configurable": {**self.config["configurable"], "thread_id": "checkpoint-test"}}
        result = app.invoke({"messages": [HumanMessage(content="Alice Fernández")]}, config)
        self.assertEqual(result["messages"][-1].content, "Alice Fernández")
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(set(saver.storage["checkpoint-test"]), {""})

    def test_escaped_unicode_quotes_and_distinct_people_round_trip(self):
        names = ['Zoë "Z" Öztürk', 'Émile Müller']
        text = json.dumps({"source": names, "username": "participant-alpha"}, ensure_ascii=True)
        config = privacy_config(self.config, dict(zip(("name", "other_name"), names)))
        result = model_call(self.model, Record).invoke([HumanMessage(content=text)], config)
        self.assertEqual(json.loads(result.text), json.loads(text))
        for name in names:
            self.assertNotIn(name, self.sent())
            self.assertNotIn(json.dumps(name)[1:-1], self.sent())
        self.assertNotEqual(*json.loads(self.requests[-1]["messages"][-1]["content"])["source"])

    def test_clinical_eponyms_drugs_values_and_dates_are_preserved(self):
        text = ("Herceptin, Taxol, Keytruda, Aspirin, Bisoprolol, Doxorubicin, Trastuzumab, "
                "Tamoxifen, Letrozole, Hodgkin lymphoma, Addison disease, Parkinson disease; "
                "May 2026, 116/67 mmHg, 49 bpm, 36.8 C, 80.9 kg, 2026-11-19T15:51:00Z.")
        result = model_call(self.model).invoke([HumanMessage(content=text)], self.config)
        self.assertEqual(self.requests[-1]["messages"][-1]["content"], text)
        self.assertEqual(result.content, text)

    def test_numeric_patient_id_is_redacted_in_fields_and_references_not_measurements(self):
        config = privacy_config({"configurable": {"user_id": "49"}}, {"patient_id": "49"})
        text = json.dumps({"user_id": "49", "subject": {"reference": "Patient/49"},
                           "value": 49, "explanation": "Heart rate 49 bpm."})
        result = model_call(self.model, Record).invoke([HumanMessage(content=text)], config)
        payload = json.loads(self.requests[-1]["messages"][-1]["content"])
        self.assertNotEqual(payload["user_id"], "49")
        self.assertNotIn("Patient/49", payload["subject"]["reference"])
        self.assertEqual(payload["value"], 49)
        self.assertEqual(payload["explanation"], "Heart rate 49 bpm.")
        self.assertEqual(json.loads(result.text), json.loads(text))

    def test_email_and_international_phone_detection_is_local(self):
        text = "Email person@example.org, telephone +44 7700 900123; 116/67 mmHg."
        result = model_call(self.model).invoke([HumanMessage(content=text)])
        self.assertNotIn("person@example.org", self.sent())
        self.assertNotIn("+44 7700 900123", self.sent())
        self.assertIn("116/67 mmHg", self.sent())
        self.assertEqual(result.content, text)

    def test_unknown_names_are_not_guessed_and_config_is_not_mutated(self):
        before = deepcopy(self.config)
        text = "My neighbour Sarah takes Taxol."
        model_call(self.model).invoke([HumanMessage(content=text)], self.config)
        self.assertEqual(self.requests[-1]["messages"][-1]["content"], text)
        self.assertEqual(self.config, before)

    def test_existing_alias_like_text_cannot_collide_with_generated_aliases(self):
        text = "PRIVATE_0 and PRIVATE__1 are literal text. Alice Fernández is the participant."
        result = model_call(self.model, Record).invoke([HumanMessage(content=text)], self.config)
        self.assertEqual(result.text, text)
        sent = self.requests[-1]["messages"][-1]["content"]
        self.assertIn("PRIVATE_0 and PRIVATE__1", sent)
        self.assertNotIn("Alice Fernández", sent)
        model_call(self.model).invoke([HumanMessage(content="Clinical question", name="PRIVATE_0")])
        self.assertNotEqual(self.requests[-1]["messages"][-1]["name"], "PRIVATE_0")

    def test_identifiers_are_request_scoped_under_concurrent_use(self):
        call = model_call(self.model, Record)
        def run(index):
            name = f"Participant-{index}"
            config = privacy_config({"configurable": {"user_id": name}}, {"name": name})
            return call.invoke([HumanMessage(content=name)], config).text
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(run, range(12)))
        self.assertEqual(results, [f"Participant-{i}" for i in range(12)])
        self.assertNotIn("Participant-", json.dumps(self.requests))

    def test_detection_failure_never_calls_the_provider(self):
        with patch("maya.framework.integrations.privacy._Pseudonyms.collect", side_effect=ValueError("invalid input")):
            with self.assertRaises(ValueError):
                model_call(self.model).invoke([HumanMessage(content="participant-alpha")], self.config)
        self.assertEqual(self.requests, [])

    def test_non_text_input_is_rejected_before_an_unprotected_request(self):
        with self.assertRaises(ValueError):
            model_call(self.model).invoke([HumanMessage(content=[{"type":"image_url", "image_url":{"url":"https://image.invalid"}}])])
        self.assertEqual(self.requests, [])


class BiomarkerPrivacyTests(unittest.TestCase):
    def test_projection_removes_identifiers_keeps_full_clinical_values_and_raw_records(self):
        from maya.framework.integrations.biomarkers import BiomarkerService
        from maya.tests.test_fhir_biomarkers import observation
        raw = observation(performer=[{"reference": "Practitioner/private-clinician"}],
                          meta={"source": "https://example.invalid/private-patient"},
                          method={"text": "Seated"}, referenceRange=[{"text":"Clinical range"}])
        records = [{"synthetic": False, "status": "ok", "observations": [{
            "id": "private-record", "patient_id": "private-patient", "source":"FHIR Observation/private-record",
            "measurement_id": "blood_pressure", "measured_at": "2026-11-19T15:51:00Z", "status":"final",
            "result": raw, "reference_range": {"ranges": raw["referenceRange"]},
        }]}]
        before = deepcopy(records)
        projected = BiomarkerService.model_results(records)
        self.assertEqual(records, before)
        text = json.dumps(projected)
        for secret in ("private-record", "private-clinician", "private-patient", "Patient/"):
            self.assertNotIn(secret, text)
        result = projected[0]["observations"][0]["result"]
        for key in ("component", "code", "effectiveDateTime", "method", "referenceRange", "status"):
            self.assertEqual(result[key], raw[key])
