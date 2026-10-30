"""Verify search text, context-preserving fallback and when rewriting is needed."""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import Mock

from langchain.agents.middleware import ModelRequest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from maya.agents.health_rag.agent import RetrieveLocalLiterature, RewrittenQuery, health_rag_prompt
from maya.framework.conversation.personalization import Preferences


class StubRetriever:
    def __init__(self, documents: list[Document] | None = None):
        self.queries: list[str] = []
        self.documents = documents if documents is not None else [
            Document(
                page_content="Global longitudinal strain below -19% predicts cardiotoxicity.",
                metadata={"title": "Cardio-oncology", "page_number": 3, "source_path": "a.pdf"},
            )
        ]

    def invoke(self, query: str) -> list[Document]:
        self.queries.append(query)
        return self.documents


class StubRewriteModel:
    def __init__(self, query: str = "", error: Exception | None = None):
        self.query = query
        self.error = error
        self.calls = 0
        self.prompts = []

    def with_structured_output(self, schema: Any) -> RunnableLambda:
        assert schema is RewrittenQuery

        def respond(prompt: Any) -> Any:
            self.calls += 1
            self.prompts.append(prompt.to_string())
            if self.error is not None:
                raise self.error
            return RewrittenQuery(query=self.query)

        return RunnableLambda(respond)


class RetrievalQueryTests(unittest.TestCase):
    def test_response_language_does_not_translate_machine_only_rewrite(self):
        model, retriever = StubRewriteModel('Clinical literature query'), StubRetriever()
        preferences = Preferences(response_language='ro')
        RetrieveLocalLiterature(retriever, model).before_model({
            'standalone_question': 'What does that mean?',
            'conversation_context': 'Previous question about doxorubicin.',
            'response_style': preferences.response_style(),
        }, None)
        self.assertEqual(model.calls, 1)
        self.assertNotIn(preferences.language_instruction(), model.prompts[0])
        self.assertEqual(retriever.queries, ['Clinical literature query'])

    def test_saved_memory_triggers_rewrite_without_clarification_or_history(self):
        memory = 'User-reported treatment: doxorubicin.'
        model = StubRewriteModel('Bisoprolol suitability after doxorubicin?')
        retriever = StubRetriever()
        update = RetrieveLocalLiterature(retriever, model).before_model({
            'standalone_question': 'Is my medication okay with my treatments?',
            'ontology_context': 'Applicable ontology context:\n' + memory,
            'ontology_memory_context': memory,
            'needs_query_rewrite': False,
        }, None)
        self.assertEqual(model.calls, 1)
        self.assertIn(memory, model.prompts[0])
        self.assertIn('doxorubicin', retriever.queries[0])
        self.assertEqual(update['retrieval_query'], retriever.queries[0])

    def test_memory_survives_timeout_empty_invalid_or_unavailable_rewriter(self):
        models = [StubRewriteModel(error=TimeoutError()), StubRewriteModel(' '),
                  Mock(with_structured_output=Mock(return_value=RunnableLambda(lambda _: {'query': []}))), None]
        for model in models:
            with self.subTest(model=model):
                retriever = StubRetriever()
                RetrieveLocalLiterature(retriever, model).before_model({
                    'standalone_question': 'What does this mean?',
                    'conversation_context': 'My current medication is bisoprolol.',
                    'biomarker_context': 'Blood pressure: 134/86 mmHg.',
                    'ontology_context': 'Full definitions and records.',
                    'ontology_memory_context': 'User-reported treatment: doxorubicin.',
                }, None)
                query = retriever.queries[0]
                for text in ('What does this mean?', 'bisoprolol', '134/86', 'doxorubicin', 'User-reported'):
                    self.assertIn(text, query)
                self.assertNotIn('Full definitions', query)

    def test_definitions_without_memory_do_not_add_a_rewrite_call(self):
        model, retriever = StubRewriteModel('Unused'), StubRetriever()
        RetrieveLocalLiterature(retriever, model).before_model({
            'standalone_question': 'What is cardiotoxicity?',
            'ontology_context': 'Applicable ontology context: definitions without records.',
            'ontology_memory_context': '',
        }, None)
        self.assertEqual(model.calls, 0)
        self.assertEqual(retriever.queries, ['What is cardiotoxicity?'])

    def test_urgent_request_bypasses_rewrite_and_retrieval_even_with_memory(self):
        model, retriever = StubRewriteModel('Unused'), StubRetriever()
        update = RetrieveLocalLiterature(retriever, model).before_model({
            'standalone_question': 'I have severe chest pain now.',
            'ontology_memory_context': 'User-reported treatment: doxorubicin.',
            'urgent_safety_bypass': True,
        }, None)
        self.assertEqual(model.calls, 0)
        self.assertEqual(retriever.queries, [])
        self.assertFalse(update['retrieval_used'])

    def test_first_turn_observations_trigger_rewrite_and_survive_rewrite_failure(self):
        state = {"standalone_question": "Explain my reading.",
                 "biomarker_context": "Synthetic blood pressure 134/86 mmHg, measured 2026-10-10."}
        for error in (None, TimeoutError()):
            with self.subTest(error=error):
                model = StubRewriteModel("Interpret blood pressure 134/86 mmHg.", error=error)
                retriever = StubRetriever()
                RetrieveLocalLiterature(retriever, model).before_model(state.copy(), None)
                self.assertEqual(model.calls, 1)
                self.assertIn("134/86", retriever.queries[0])
                request = ModelRequest(model=Mock(), messages=[], state=state)
                prompt = health_rag_prompt(request)
                self.assertIn(state["biomarker_context"], prompt)

    def test_response_personality_reaches_health_prompt_with_or_without_evidence(self):
        style = Preferences(personality_id="concise").response_style()
        for citations in ([], [{"title": "Source", "excerpt": "Evidence", "page_number": 0}]):
            with self.subTest(citations=citations):
                request = ModelRequest(model=Mock(), messages=[], state={"response_style": style, "citations": citations})
                prompt = health_rag_prompt(request)
                self.assertIn(style, prompt)
                self.assertIn("Ground medical explanations only in the retrieved local literature", prompt)
                self.assertIn("If the retrieved literature is absent or insufficient", prompt)

    def test_single_turn_question_is_searched_verbatim_with_no_model_call(self):
        retriever, model = StubRetriever(), StubRewriteModel("should not be used")
        middleware = RetrieveLocalLiterature(retriever, model)

        update = middleware.before_model(
            {"standalone_question": "What is cardiotoxicity?"}, None
        )

        self.assertEqual(retriever.queries, ["What is cardiotoxicity?"])
        self.assertEqual(model.calls, 0)
        self.assertEqual(update["retrieval_query"], "What is cardiotoxicity?")

    def test_first_turn_clarification_reply_is_rewritten_without_prior_history(self):
        retriever, model = StubRetriever(), StubRewriteModel('Medication suitability after the reported treatment')
        RetrieveLocalLiterature(retriever, model).before_model({
            'standalone_question': 'Original question, clarification question and user reply',
            'needs_query_rewrite': True,
        }, None)
        self.assertEqual(model.calls, 1)
        self.assertEqual(retriever.queries, ['Medication suitability after the reported treatment'])

    def test_multi_turn_question_is_rewritten_before_retrieval(self):
        retriever = StubRetriever()
        model = StubRewriteModel(
            "What is the significance of a global longitudinal strain of -15% after "
            "cyclophosphamide?"
        )
        middleware = RetrieveLocalLiterature(retriever, model)

        update = middleware.before_model(
            {
                "standalone_question": "earlier turns\n\nCurrent user question: is that bad?",
                "conversation_context": "assistant: which test?\nuser: GLS -15%",
            },
            None,
        )

        self.assertEqual(model.calls, 1)
        self.assertEqual(
            retriever.queries,
            [
                "What is the significance of a global longitudinal strain of -15% after "
                "cyclophosphamide?"
            ],
        )
        self.assertEqual(update["retrieval_query"], retriever.queries[0])

    def test_rewrite_failure_falls_back_to_the_supplied_text(self):
        """A degraded query is recoverable; a failed turn is not."""

        retriever = StubRetriever()
        model = StubRewriteModel(error=RuntimeError("rate limited"))
        middleware = RetrieveLocalLiterature(retriever, model)

        update = middleware.before_model(
            {"standalone_question": "context\n\nquestion", "conversation_context": "prior"},
            None,
        )

        self.assertEqual(retriever.queries, ["prior\n\ncontext\n\nquestion"])
        self.assertEqual(update["retrieval_query"], "prior\n\ncontext\n\nquestion")

    def test_empty_rewrite_falls_back_to_the_supplied_text(self):
        retriever, model = StubRetriever(), StubRewriteModel("   ")
        middleware = RetrieveLocalLiterature(retriever, model)

        middleware.before_model(
            {"standalone_question": "context\n\nquestion", "conversation_context": "prior"},
            None,
        )

        self.assertEqual(retriever.queries, ["prior\n\ncontext\n\nquestion"])

    def test_missing_model_degrades_to_verbatim_retrieval(self):
        """The middleware must stay constructible without a model."""

        retriever = StubRetriever()
        middleware = RetrieveLocalLiterature(retriever)

        middleware.before_model(
            {"standalone_question": "context\n\nquestion", "conversation_context": "prior"},
            None,
        )

        self.assertEqual(retriever.queries, ["prior\n\ncontext\n\nquestion"])

    def test_existing_query_is_reused_rather_than_rebuilt(self):
        """before_model runs ahead of every model call; the rewrite is paid for once."""

        retriever, model = StubRetriever(), StubRewriteModel("fresh")
        middleware = RetrieveLocalLiterature(retriever, model)

        middleware.before_model(
            {
                "standalone_question": "context\n\nquestion",
                "conversation_context": "prior",
                "retrieval_query": "already built",
            },
            None,
        )

        self.assertEqual(model.calls, 0)
        self.assertEqual(retriever.queries, ["already built"])

    def test_question_falls_back_to_the_last_human_message(self):
        from langchain_core.messages import AIMessage, HumanMessage

        retriever, model = StubRetriever(), StubRewriteModel()
        middleware = RetrieveLocalLiterature(retriever, model)

        middleware.before_model(
            {
                "messages": [
                    HumanMessage(content="first"),
                    AIMessage(content="reply"),
                    HumanMessage(content="Is my heart at risk?"),
                ]
            },
            None,
        )

        self.assertEqual(retriever.queries, ["Is my heart at risk?"])

    def test_no_question_retrieves_nothing_and_sets_no_query(self):
        retriever, model = StubRetriever(), StubRewriteModel()
        middleware = RetrieveLocalLiterature(retriever, model)

        update = middleware.before_model({"messages": []}, None)

        self.assertEqual(update, {"citations": []})
        self.assertEqual(retriever.queries, [])
        self.assertEqual(model.calls, 0)

    def test_citation_shape_is_unchanged(self):
        """Downstream consumers read these keys; the rewrite must not disturb them."""

        retriever = StubRetriever(
            [
                Document(
                    page_content="  spaced   text  ",
                    metadata={"title": "T", "page_number": 1, "source_path": "p.pdf"},
                )
            ]
        )
        middleware = RetrieveLocalLiterature(retriever, StubRewriteModel())

        update = middleware.before_model({"standalone_question": "q"}, None)

        self.assertEqual(
            update["citations"],
            [{"title": "T", "source_path": "p.pdf", "page_number": 1, "excerpt": "spaced text"}],
        )

    def test_long_excerpt_is_truncated_on_a_word_boundary(self):
        retriever = StubRetriever(
            [Document(page_content="word " * 200, metadata={"title": "T"})]
        )
        middleware = RetrieveLocalLiterature(retriever, StubRewriteModel())

        excerpt = middleware.before_model({"standalone_question": "q"}, None)["citations"][0][
            "excerpt"
        ]

        self.assertTrue(excerpt.endswith("..."))
        self.assertLessEqual(len(excerpt), 500)

    def test_at_most_three_citations_are_kept(self):
        retriever = StubRetriever(
            [Document(page_content=f"doc {i}", metadata={"title": str(i)}) for i in range(6)]
        )
        middleware = RetrieveLocalLiterature(retriever, StubRewriteModel())

        update = middleware.before_model({"standalone_question": "q"}, None)

        self.assertEqual(len(update["citations"]), 3)


if __name__ == "__main__":
    unittest.main()
