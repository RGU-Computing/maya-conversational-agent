"""Live supervisor checks for configurable HealthRAG.

Run: python -m maya.tests.live_health_config_scenarios [case ...]
Uses configured models/literature, synthetic observations and isolated RAM memory.
No participant database writes, audio playback or ICE actions.
"""

import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from unittest.mock import patch

from dotenv import load_dotenv


def run(selected=(), *, service=None):
    load_dotenv()
    os.environ['LANGSMITH_TRACING'] = 'false'
    os.environ['LANGCHAIN_TRACING_V2'] = 'false'
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.runnables import RunnableLambda
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
    from langgraph.types import Command
    from maya.agents.health_rag.config import load_health_rag_config
    from maya.agents.health_rag.agent import HealthRagHandle, RetrieveLocalLiterature
    from maya.framework.memory.ontology import OntologyMemoryService, OntologyMemoryExtraction, OntologyMemoryVerification
    from maya.framework.conversation.personalization import SessionConfig
    from maya.framework.supervisor import factory
    from maya.tests.test_biomarkers import demo_service

    store, service, workers, report = InMemoryStore(), service or demo_service(), [], []
    memory_events = []
    output = Path('output/health_config_live' + ('_' + '_'.join(selected) if selected else '') + '.json')
    output.parent.mkdir(exist_ok=True)

    def build(specs=None):
        def traced_memory_model(schema):
            def create(model):
                structured = model.with_structured_output(schema, method='function_calling')
                def invoke(messages, config):
                    result = structured.invoke(messages, config)
                    memory_events.append({'stage': schema.__name__, 'input': messages[-1].content,
                                          'result': result.model_dump()})
                    return result
                return RunnableLambda(invoke)
            return create

        def memory_factory(**kwargs):
            worker = OntologyMemoryService(**kwargs, register_shutdown=False,
                extractor_factory=traced_memory_model(OntologyMemoryExtraction),
                verifier_factory=traced_memory_model(OntologyMemoryVerification))
            workers.append(worker)
            return worker
        with patch.object(factory, 'get_postgres_memory', return_value=(store, InMemorySaver())), \
                patch.object(factory, 'OntologyMemoryService', side_effect=memory_factory), \
                patch.object(factory, 'load_agent_specs', return_value=specs or factory.load_agent_specs()):
            return factory.build_supervisor(biomarker_service=service)

    graph = build()

    def drain():
        for worker in workers:
            with worker._condition:
                assert worker._condition.wait_for(lambda: not worker._scheduled_jobs, timeout=120), 'Memory timeout'

    def turn(graph, config, value, *, rewrite_model=None):
        start = monotonic()
        searches = []
        rewrite = RetrieveLocalLiterature._retrieval_query
        def traced_query(middleware, state, question):
            target = middleware if rewrite_model is None else RetrieveLocalLiterature(
                middleware.retriever, rewrite_model, middleware.settings)
            query = rewrite(target, state, question)
            searches.append({'question': question, 'memory': state.get('ontology_memory_context', ''),
                             'query': query})
            return query
        with patch.object(service.provider, 'fetch_latest', wraps=service.provider.fetch_latest) as fetch, \
                patch.object(RetrieveLocalLiterature, '_retrieval_query', traced_query):
            result = graph.invoke(value, config)
        message = result['messages'][-1]
        entry = {'seconds': round(monotonic() - start, 2), 'route': result.get('route_type'),
                 'interrupts': [i.value for i in graph.get_state(config).interrupts],
                 'answer': message.content if isinstance(message, AIMessage) else '',
                 'metadata': message.additional_kwargs, 'lookup_calls': fetch.call_count,
                 'lookup_measurements': [call.args[1] for call in fetch.call_args_list],
                 'selected_models': result.get('selected_ontology_models', []),
                 'question_subject': result.get('question_subject'),
                 'observations': result.get('biomarker_results', []), 'searches': searches}
        drain()
        return entry

    def ask(question, user='health-config', *, target=None, mapped=True):
        if mapped:
            service.config.patient_map[user] = 'synthetic-patient'
        config = SessionConfig(user_id=user).graph_config()
        return config, turn(target or graph, config, {'messages': [HumanMessage(content=question)]})

    def answered(entry):
        presentation = entry['metadata'].get('presentation', {})
        refs = presentation.get('references', [])
        speech = presentation.get('speech_text', '')
        return (entry['route'] == 'health_rag' and not entry['interrupts'] and bool(refs)
                and all(ref['file_name'] not in speech for ref in refs if ref['file_name']))

    def save(name, checks, **evidence):
        entry = {'case': name, 'passed': all(checks.values()), 'checks': checks, **evidence}
        report.append(entry)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        print(json.dumps({'case': name, 'passed': entry['passed'], 'checks': checks}), flush=True)

    def enabled(name):
        if not selected or name in selected:
            print('Running ' + name, flush=True)
            return True
        return False

    try:
        if enabled('general_cache_config'):
            question = 'In general, what is cardiotoxicity after cancer treatment?'
            _, first = ask(question, 'health-config-cache')
            _, cached = ask(question, 'health-config-cache')
            with TemporaryDirectory() as folder:
                settings = load_health_rag_config().model_dump()
                marker = 'Here is what the available evidence says.'
                settings['prompts']['answer'] += '\nBegin the answer with this exact sentence: ' + marker
                path = Path(folder) / 'health.json'
                path.write_text(json.dumps(settings))
                specs = copy.deepcopy(factory.load_agent_specs())
                specs['health_rag']['config_path'] = str(path)
                changed_graph = build(specs)
                _, changed = ask(question, 'health-config-cache', target=changed_graph)
            save('general_cache_config', {
                'general_answer': answered(first), 'no_observation_lookup': first['lookup_calls'] == 0,
                'cache_matches_effective_context': cached['metadata'].get('workflow_trace', [{}])[0].get('memory_lookup') == (
                    'exact' if first['metadata'].get('workflow_trace', [{}])[0].get('memory_key') ==
                    cached['metadata'].get('workflow_trace', [{}])[0].get('memory_key') else 'miss'),
                'changed_config_retrieves_again': changed['metadata'].get('workflow_trace', [{}])[0].get('retrieval_used') is True,
                'partner_prompt_changes_answer': marker in changed['answer'],
            }, first=first, cached=cached, changed=changed)

        if enabled('treatment_memory'):
            question = "I'm taking bisoprolol for my blood pressure. Is it okay because of my treatments?"
            config, first = ask(question, 'health-config-treatment')
            response = turn(graph, config, Command(resume='I finished doxorubicin chemotherapy six months ago.')) if first['interrupts'] else first
            assess = HealthRagHandle.assess_clarification
            def without_memory_labels(handle, *args, **kwargs):
                result = assess(handle, *args, **kwargs)
                # Reproduce the original trigger failure even if this model run labels correctly.
                for _, slot in result.picot:
                    if slot.source == 'ontology_profile':
                        slot.source = 'none'
                return result
            with patch.object(HealthRagHandle, 'assess_clarification', without_memory_labels):
                _, recalled = ask(question, 'health-config-treatment')
            pending = recalled if recalled['metadata'].get('memory_status') == 'pending' else None
            if pending:
                # load_context schedules recovery. turn() drains that worker before retrying.
                with patch.object(HealthRagHandle, 'assess_clarification', without_memory_labels):
                    _, recalled = ask(question, 'health-config-treatment')
            _, friend = ask('My friend takes bisoprolol. Is it okay with her cancer treatments?', 'health-config-treatment')
            _, isolated = ask(question, 'health-config-other-user')
            records = workers[0].load_records('health-config-treatment', ['cancer_history'])
            save('treatment_memory', {
                'asks_treatment': 'treatment' in str(first['interrupts']).lower(),
                'resume_answers': answered(response), 'treatment_stored': 'doxorubicin' in json.dumps(records).lower(),
                'new_thread_uses_memory': answered(recalled),
                'retrieval_uses_saved_treatment': 'doxorubicin' in recalled['metadata'].get('workflow_trace', [{}])[0].get('retrieval_query', '').lower(),
                'memory_reaches_retrieval': bool(recalled['searches']) and 'doxorubicin' in recalled['searches'][0]['memory'].lower(),
                'friend_does_not_inherit_history': bool(friend['interrupts']) and friend['lookup_calls'] == 0
                    and friend['question_subject'] != 'current_user',
                'different_user_needs_own_treatment': bool(isolated['interrupts']),
            }, first=first, resumed=response, pending=pending, recalled=recalled, friend=friend,
                 isolated=isolated, records=records, memory_events=list(memory_events))

        if enabled('memory_rewrite_failures'):
            from maya.tests.test_health_rag_retrieval import StubRewriteModel
            question = "I'm taking bisoprolol for my blood pressure. Is it okay because of my treatments?"
            user = 'health-config-rewrite-failure'
            config, first = ask(question, user)
            response = turn(graph, config, Command(resume='I finished doxorubicin chemotherapy six months ago.')) if first['interrupts'] else first
            cases = []
            for name, rewriter in [('timeout', StubRewriteModel(error=TimeoutError('Injected rewrite timeout'))),
                                   ('empty', StubRewriteModel(' '))]:
                # Each fault must reach retrieval rather than reuse the previous answer.
                namespace = ('health_rag', user, 'qa_history')
                for item in store.search(namespace):
                    store.delete(namespace, item.key)
                result = turn(graph, SessionConfig(user_id=user).graph_config(),
                              {'messages': [HumanMessage(content=question)]}, rewrite_model=rewriter)
                query = result['metadata'].get('workflow_trace', [{}])[0].get('retrieval_query', '')
                cases.append({'fault': name, 'result': result, 'checks': {
                    'rewrite_attempted_once': rewriter.calls == 1,
                    'memory_preserved_in_search': 'doxorubicin' in query.lower() and 'user_reported' in query,
                    'question_preserved': question in query,
                    'definitions_excluded_from_search': 'Applicable ontology context:' not in query,
                    'retrieved_answer_and_display_only_references': answered(result),
                }})
            save('memory_rewrite_failures', {'setup_answers': answered(response),
                **{case['fault']: all(case['checks'].values()) for case in cases}}, first=first, response=response, cases=cases)

        if enabled('biomarker_clarification'):
            question = 'Please interpret my latest blood pressure and explain whether it is safe given the cancer treatment I received.'
            config, first = ask(question, 'health-config-biomarker')
            response = turn(graph, config, Command(resume='Doxorubicin, completed six months ago.')) if first['interrupts'] else first
            query = response['metadata'].get('workflow_trace', [{}])[0].get('retrieval_query', '')
            save('biomarker_clarification', {
                'fetches_before_clarification': first['lookup_calls'] == 1 and bool(first['interrupts']),
                'asks_treatment': 'treatment' in str(first['interrupts']).lower(),
                'does_not_refetch_on_resume': response['lookup_calls'] == 0,
                'answers_with_evidence': answered(response),
                'query_contains_observation_and_treatment': '134' in query and 'doxorubicin' in query.lower(),
            }, first=first, resumed=response)

        if enabled('unknown_test'):
            config, first = ask('My cardiac blood test was high. What does that mean?', 'health-config-unknown')
            response = turn(graph, config, Command(resume="I don't know the test name or value.")) if first['interrupts'] else first
            save('unknown_test', {
                'asks_which_test': bool(first['interrupts']) and 'test' in str(first['interrupts']).lower(),
                'does_not_guess_lookup': first['lookup_calls'] == 0,
                'unknown_reply_completes': bool(response['answer']) and not response['interrupts'],
            }, first=first, resumed=response)

        if enabled('spanish'):
            _, result = ask('¿Qué es la cardiotoxicidad, en general?', 'health-config-spanish')
            save('spanish', {'answers_with_evidence': answered(result), 'no_lookup': result['lookup_calls'] == 0,
                             'spanish_response': 'cardiotoxicidad' in result['answer'].lower()}, result=result)

        if enabled('urgent'):
            _, result = ask('I have chest pain and shortness of breath right now. What should I do?', 'health-config-urgent')
            save('urgent', {'no_clarification': not result['interrupts'], 'no_lookup': result['lookup_calls'] == 0,
                            'no_retrieval_delay': result['metadata'].get('workflow_trace', [{}])[0].get('retrieval_used') is False,
                            'urgent_guidance': any(word in result['answer'].lower() for word in ('emergency', 'immediate', 'urgent'))}, result=result)

        if enabled('missing_observation'):
            _, result = ask('Please interpret my latest blood pressure using my recorded measurement.',
                            'health-config-unmapped', mapped=False)
            save('missing_observation', {'asks_for_unavailable_measurement': bool(result['interrupts']),
                                         'no_other_patient_lookup': result['lookup_calls'] == 0}, result=result)

        if enabled('rewrite_context'):
            config, first = ask('What heart monitoring is generally used after doxorubicin?', 'health-config-context')
            result = turn(graph, config, {'messages': [HumanMessage(content='And how long after that treatment is monitoring relevant?')]})
            query = result['metadata'].get('workflow_trace', [{}])[0].get('retrieval_query', '')
            save('rewrite_context', {'first_answer': answered(first), 'followup_answer': answered(result),
                                     'resolves_treatment_for_search': 'doxorubicin' in query.lower()}, first=first, result=result)

        if enabled('empty_evidence'):
            # Deliberately inject an empty retrieval result; routing and answer generation stay live.
            with patch('maya.agents.health_rag.agent.build_vector_store') as vector:
                vector.return_value.as_retriever.return_value.invoke.return_value = []
                empty_graph = build()
            _, result = ask('In general, what is cardiotoxicity?', 'health-config-empty', target=empty_graph)
            presentation = result['metadata'].get('presentation', {})
            save('empty_evidence', {
                'health_route_completes': result['route'] == 'health_rag' and not result['interrupts'],
                'no_invented_references': presentation.get('references') == [],
                'states_evidence_limit': any(word in result['answer'].lower() for word in ('literature', 'evidence')),
            }, result=result)
    finally:
        for worker in workers:
            worker.close()
    assert all(entry['passed'] for entry in report), f'Review {output}'
    return report


if __name__ == '__main__':
    import sys
    run(sys.argv[1:])
