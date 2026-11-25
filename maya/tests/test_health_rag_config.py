"""Partner configuration reaches each model stage without parsing prompt text as code."""

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from maya.tests.model_stubs import model_call_stub

from langchain.agents.middleware import ModelRequest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from maya.agents.config import _load_specs
from maya.agents.health_rag import agent
from maya.agents.health_rag.config import HealthRagConfig, load_health_rag_config
from maya.tests.test_health_rag_clarification import StubStructuredModel
from maya.tests.test_health_rag_retrieval import StubRetriever


class HealthConfigTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("maya.agents.health_rag.agent.model_call", side_effect=model_call_stub))
        self.settings = load_health_rag_config()

    def test_json_and_jsonl_preserve_urls_braces_and_comment_like_prompt_text(self):
        prompt = 'Use https://example.org/a, literal {context}, /* text */ and [one,].'
        entries = [{"health_rag": {"config_path": "health.json"}}, {"exercise": {"prompt": prompt}}]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "agents.jsonl"
            for source in (json.dumps({**entries[0], **entries[1]}), '\n'.join(map(json.dumps, entries))):
                path.write_text(source)
                specs = _load_specs(path)
                self.assertEqual(specs["exercise"]["prompt"], prompt)
                self.assertEqual(specs["health_rag"]["config_path"], str((Path(folder) / "health.json").resolve()))

    def test_invalid_or_duplicate_agent_config_fails_instead_of_changing_text(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "agents.json"
            for text in ('{"health_rag": {},}', '{"health_rag": {}}\n{"health_rag": {}}', ''):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    _load_specs(path)

    def test_missing_blank_unknown_and_invalid_settings_fail(self):
        for edit in (
            lambda p: p['prompts'].pop('answer'),
            lambda p: p['prompts'].update(answer=' '),
            lambda p: p['prompts'].update(typo='ignored?'),
            lambda p: p['retrieval'].update(k=0),
            lambda p: p['retrieval'].update(score_threshold=1.1),
        ):
            payload = self.settings.model_dump()
            edit(payload)
            with self.assertRaises(ValidationError):
                HealthRagConfig.model_validate(payload)

    def test_custom_config_is_used_by_build_answer_rewrite_and_retrieval(self):
        payload = self.settings.model_dump()
        payload['prompts']['answer'] = 'Partner answer instruction with {literal} and https://example.org.'
        payload['prompts']['query_rewrite'] = 'Partner rewrite instruction with {literal}.'
        payload['retrieval'].update(k=7, citation_limit=1, excerpt_chars=24, score_threshold=0.4)
        settings = HealthRagConfig.model_validate(payload)
        store = Mock()
        retriever = StubRetriever([Document(page_content='Evidence ' * 20), Document(page_content='Unused')])
        store.as_retriever.return_value = retriever
        captured = []
        model = Mock()
        model.with_structured_output.return_value = RunnableLambda(
            lambda prompt: captured.append(prompt.to_messages()) or agent.RewrittenQuery(query='rewritten'))
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'health.json'
            path.write_text(settings.model_dump_json())
            vector_config = SimpleNamespace(
                backend='pgvector', collection_name='test', embedding_backend='deterministic', embedding_model='test',
            )
            with patch.object(agent, 'load_rag_vector_config', return_value=vector_config), \
                    patch.object(agent, 'load_chunk_snapshot', return_value=[]), \
                    patch.object(agent, 'build_embeddings'), \
                    patch.object(agent, 'build_vector_store', return_value=store), \
                    patch.object(agent, 'get_chat_model', return_value=model), \
                    patch.object(agent, 'create_agent') as create:
                handle = agent.build({'config_path': str(path)})
            retrieval, prompt, privacy = create.call_args.kwargs['middleware']
            update = retrieval.before_model({'standalone_question': 'Current question',
                                             'conversation_context': 'History once'}, None)
            rendered = prompt.wrap_model_call(ModelRequest(model=Mock(), messages=[], state=update),
                                              lambda r: r.system_message.content)
        self.assertEqual(handle.config, settings)
        store.as_retriever.assert_called_once_with(search_type='similarity_score_threshold',
                                                   search_kwargs={'k': 7, 'score_threshold': 0.4})
        self.assertIn(settings.prompts.answer, rendered)
        self.assertIn(settings.prompts.query_rewrite, captured[0][0].content)
        self.assertEqual(captured[0][1].content.count('History once'), 1)
        self.assertEqual(len(update['citations']), 1)
        self.assertLessEqual(len(update['citations'][0]['excerpt']), 24)

    def test_custom_assessment_and_wording_are_not_interpreted_as_templates(self):
        settings = self.settings.model_copy(update={'prompts': self.settings.prompts.model_copy(update={
            'assessment': 'Assess using {literal}. https://example.org/a',
            'clarification': 'Ask gently using {literal}.'})})
        handle = agent.HealthRagHandle(name='health_rag', agent=None, ontology_block='Partner ontology', config=settings)
        model = StubStructuredModel(agent.DisambiguationDecision(needs_clarification=True,
            missing_ontology_fields=['treatment'], material_missing_fields=['treatment']), wording='Which treatment?')
        handle.assess_clarification('Is it suitable?', model=model)
        self.assertIn(settings.prompts.assessment, model.inputs[0].to_messages()[0].content)
        self.assertIn('Partner ontology', model.inputs[0].to_messages()[0].content)
        self.assertIn(settings.prompts.clarification, model.inputs[1].to_messages()[0].content)

    def test_configured_repair_instruction_reaches_the_repair_call(self):
        settings = self.settings.model_copy(update={'prompts': self.settings.prompts.model_copy(
            update={'assessment_repair': 'Partner repair: return all rules, preserving {literal}.'})})
        handle = agent.HealthRagHandle(name='health_rag', agent=None, ontology_block='Partner ontology',
            ontology_entity_ids=frozenset({'scope'}), ontology_rule_counts={'scope': 1}, config=settings)
        responses = iter([
            agent.DisambiguationDecision(needs_clarification=False),
            agent.DisambiguationDecision(needs_clarification=False, ontology_rule_checks=[
                agent.OntologyRuleCheck(entity_id='scope', rule_index=1)]),
        ])
        captured = []
        model = Mock()
        model.with_structured_output.return_value = RunnableLambda(
            lambda prompt: captured.append(prompt.to_messages()) or next(responses))
        handle.assess_clarification('A general question', model=model)
        self.assertEqual(len(captured), 2)
        self.assertNotIn(settings.prompts.assessment_repair, captured[0][0].content)
        self.assertIn(settings.prompts.assessment_repair, captured[1][0].content)

    def test_personality_wording_is_configurable_for_each_existing_id(self):
        from maya.framework.conversation.personalization import PERSONALITIES, Preferences, PersonalityPrompts
        with patch.dict(PERSONALITIES, {key: f'Partner wording for {key}' for key in PERSONALITIES}):
            for key, instruction in PERSONALITIES.items():
                self.assertIn(instruction, Preferences(personality_id=key).response_style())
        with self.assertRaises(ValidationError):
            PersonalityPrompts.model_validate({'default': '', 'supportive': '', 'concise': 'Brief'})

    def test_changed_instructions_ontology_records_or_style_invalidate_cache(self):
        handle = agent.HealthRagHandle(name='health_rag', agent=None, ontology_block='Ontology')
        original = handle.cache_scope('supportive', 'User record')
        self.assertEqual(original, replace(handle).cache_scope('supportive', 'User record'))
        for field in type(self.settings.prompts).model_fields:
            config = self.settings.model_copy(update={'prompts': self.settings.prompts.model_copy(
                update={field: getattr(self.settings.prompts, field) + ' Changed.'})})
            self.assertNotEqual(original, replace(handle, config=config).cache_scope('supportive', 'User record'))
        self.assertNotEqual(original, replace(handle, ontology_block='New ontology').cache_scope('supportive', 'User record'))
        self.assertNotEqual(original, replace(handle, literature_revision='Updated').cache_scope('supportive', 'User record'))
        self.assertNotEqual(original, handle.cache_scope('concise', 'User record'))
        self.assertNotEqual(original, handle.cache_scope('supportive', 'Changed record'))

    def test_stale_agent_prompt_field_is_rejected_before_initializing_retrieval(self):
        with patch.object(agent, 'build_embeddings') as embeddings:
            with self.assertRaisesRegex(ValueError, 'Unsupported HealthRAG'):
                agent.build({'prompt': 'This used to be silently ignored'})
            embeddings.assert_not_called()

    def test_empty_evidence_ends_real_agent_graph_without_a_model_answer(self):
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        from langchain_core.messages import HumanMessage, AIMessage
        model = FakeMessagesListChatModel(responses=[AIMessage(content='Must not be generated')])
        graph = agent.create_agent(model, tools=[],
            middleware=[agent.RetrieveLocalLiterature(StubRetriever([]), settings=self.settings)],
            state_schema=agent.HealthRagState)
        with patch.object(FakeMessagesListChatModel, '_generate', side_effect=AssertionError('Unexpected generation')):
            result = graph.invoke({'messages': [HumanMessage(content='General health question')]})
        self.assertEqual(result['messages'][-1].content, self.settings.messages.no_evidence)
        self.assertEqual(result['citations'], [])

    def test_urgent_path_skips_retrieval_and_uses_configured_safety_prompt(self):
        retriever = Mock()
        state = {'urgent_safety_bypass': True, 'standalone_question': 'Urgent concern'}
        update = agent.RetrieveLocalLiterature(retriever, settings=self.settings).before_model(state, None)
        retriever.invoke.assert_not_called()
        self.assertFalse(update['retrieval_used'])
        self.assertNotIn('jump_to', update)
        prompt = agent.health_rag_prompt(ModelRequest(model=Mock(), messages=[], state=state), self.settings)
        self.assertIn(self.settings.prompts.urgent_answer.split("{emergency_contact}")[0].rstrip(), prompt)
        self.assertNotIn(agent.EMERGENCY_CONTACT_MARKER, prompt)
        self.assertNotIn("{emergency_contact}", prompt)
        self.assertNotIn(self.settings.prompts.answer, prompt)


if __name__ == '__main__':
    unittest.main()
