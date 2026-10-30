"""Real supervisor/model/retrieval language checks; fixture tools and RAM memory.

Run explicitly: python -m maya.tests.live_response_language_scenarios ro [--edges]
Uses the runner's load_session(JSON) path. Never contacts Phyxio/FHIR or plays audio.
Full evidence stays in ignored output; language quality still needs native review.
"""

import json
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from unittest.mock import patch

from dotenv import load_dotenv


def run(language, *, edges=False, medication=False, only_medication=False, label='', output_path=None):
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / '.env')
    os.environ['LANGSMITH_TRACING'] = 'false'
    os.environ['LANGCHAIN_TRACING_V2'] = 'false'
    os.environ['MAYA_ENABLE_PHYXIO_IO'] = '0'
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
    from langgraph.types import Command
    from maya.framework.conversation.personalization import load_session, RESPONSE_LANGUAGES
    from maya.framework.memory.ontology import OntologyMemoryService
    from maya.framework.supervisor import factory
    from maya.agents.health_rag.agent import HealthRagHandle, RetrieveLocalLiterature
    from maya.agents.phyxio_exercise_agent import agent as exercise, phyxio_sessions
    from maya.tests.test_fhir_biomarkers import configured_service, observation, records
    from maya.tests.test_phyxio_sessions import configured_service as exercise_config, history_reader
    from maya.tests.test_phyxio_offline_integration import make_service
    from maya.tests.test_health_rag_retrieval import StubRetriever

    assert language in RESPONSE_LANGUAGES.languages
    suffix = f'_{label}' if label else ''
    output = Path(output_path) if output_path else root / 'output' / f'response_language_{language}{"_edges" if edges else ""}{suffix}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'language': language, 'language_name': RESPONSE_LANGUAGES.languages[language],
              'scope': 'Real models, supervisor, ontology memory and literature retrieval; fixture SDK/Phyxio, RAM storage, no audio',
              'cases': []}
    workers, assessments, searches = [], [], []
    store = InMemoryStore()
    def memory_factory(**kwargs):
        worker = OntologyMemoryService(**kwargs, register_shutdown=False)
        workers.append(worker)
        return worker

    assess = HealthRagHandle.assess_clarification
    def traced_assess(handle, *args, **kwargs):
        decision = assess(handle, *args, **kwargs)
        assessments.append({'question': args[0], 'context': kwargs.get('ontology_context'),
                            'instruction': kwargs.get('response_style'), 'decision': decision.model_dump()})
        return decision

    rewrite = RetrieveLocalLiterature._retrieval_query
    def traced_query(middleware, state, question):
        query = rewrite(middleware, state, question)
        searches.append({'question': question, 'query': query,
                         'memory': state.get('ontology_memory_context'),
                         'observations': state.get('biomarker_context')})
        return query

    def drain():
        for worker in workers:
            with worker._condition:
                if not worker._condition.wait_for(lambda: not worker._scheduled_jobs, timeout=120):
                    raise TimeoutError('Isolated ontology-memory worker did not finish')

    def save(name, checks, **evidence):
        entry = {'case': name, 'passed': all(checks.values()), 'checks': checks, **evidence}
        report['cases'].append(entry)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        print(json.dumps({'language': language, 'case': name, 'passed': entry['passed'], 'checks': checks}), flush=True)

    def trace(entry):
        return entry['metadata'].get('workflow_trace', [{}])[0]

    def answered(entry):
        refs = entry['metadata'].get('presentation', {}).get('references', [])
        speech = entry['metadata'].get('presentation', {}).get('speech_text', '')
        return (entry['route'] == 'health_rag' and not entry['interrupts'] and bool(entry['answer']) and bool(refs)
                and all(ref['file_name'] not in speech for ref in refs if ref['file_name']))

    with TemporaryDirectory() as folder:
        directory = Path(folder)
        biomarkers = configured_service(directory)
        biomarkers.provider.synthetic = True
        biomarkers.config.patient_map = {}
        biomarkers.provider.client.get.return_value = records(observation())
        exercise_config(directory)
        mirror = make_service()
        mirror.session_history_reader.return_value = history_reader()

        def session(case, code=language, *, user=None, omitted=False):
            user = user or f'language-{language}-{case}'
            path = directory / 'session.json'
            preferences = {'personality_id': 'supportive'}
            if not omitted:
                preferences['response_language'] = code
            path.write_text(json.dumps({'user_id': user, 'preferences': preferences}))
            loaded = load_session(path)
            biomarkers.config.patient_map[user] = 'synthetic-patient'
            return loaded, loaded.graph_config()

        def turn(config, value):
            start = monotonic()
            assessments.clear(); searches.clear()
            biomarkers.provider.client.get.reset_mock()
            mirror.reset_mock()
            fetched = []
            fetch = biomarkers.fetch
            def observed_fetch(*args, **kwargs):
                outcome = fetch(*args, **kwargs)
                fetched.append(outcome)
                return outcome
            with patch.object(biomarkers, 'fetch', side_effect=observed_fetch):
                result = graph.invoke(value, config)
            message = result['messages'][-1]
            entry = {'seconds': round(monotonic()-start, 2), 'route': result.get('route_type'),
                     'interrupts': [i.value for i in graph.get_state(config).interrupts],
                     'answer': message.content if isinstance(message, AIMessage) else '',
                     'metadata': message.additional_kwargs, 'selected_models': result.get('selected_ontology_models', []),
                     'ontology_context': result.get('ontology_context', ''), 'subject': result.get('question_subject'),
                     # A health-node interrupt can occur before its local lookup is
                     # committed to graph state. Verify the actual service result too.
                     'observations': result.get('biomarker_results') or (fetched[-1] if fetched else []),
                     'checkpoint_observations': result.get('biomarker_results', []),
                     'sdk_calls': [{'resource': call.args[0], **call.kwargs} for call in biomarkers.provider.client.get.call_args_list],
                     'exercise_calls': [str(call) for call in mirror.mock_calls],
                     'assessments': list(assessments), 'searches': list(searches)}
            drain()
            return entry

        def ask(case, question, **kwargs):
            loaded, config = session(case, **kwargs)
            return loaded, config, turn(config, {'messages': [HumanMessage(content=question)]})

        def medication_case():
            _, _, medication = ask('medication', 'My medication is bisoprolol 2.5 mg daily, my blood pressure is 134/82 mmHg, '
                'and I finished doxorubicin six months ago. Explain why my care team must review suitability rather '
                'than assuming this is definitely safe. Keep those medication names, dose and reading in the explanation.')
            save('medication_values_uncertainty', {'answer': answered(medication),
                'names_and_units': all(re.search(r'(?<!\w)' + re.escape(token) + r'(?!\w)', medication['answer'], re.IGNORECASE)
                                       for token in ('bisoprolol', 'doxorubicin', 'mg', 'mmHg', '134', '82')),
                'dose': '2.5' in medication['answer'] or '2,5' in medication['answer']}, result=medication)

        try:
            with patch.object(factory, 'get_postgres_memory', return_value=(store, InMemorySaver())), \
                    patch.object(factory, 'OntologyMemoryService', side_effect=memory_factory), \
                    patch.object(exercise, 'get_phyxio_service', return_value=mirror), \
                    patch.object(phyxio_sessions, 'get_phyxio_service', return_value=mirror), \
                    patch.dict(os.environ, {'MAYA_EXERCISE_SESSION_CONFIG': str(directory / 'exercise.json')}), \
                    patch.object(HealthRagHandle, 'assess_clarification', traced_assess), \
                    patch.object(RetrieveLocalLiterature, '_retrieval_query', traced_query):
                graph = factory.build_supervisor(biomarker_service=biomarkers)
                if only_medication:
                    medication_case()
                    report['passed'] = all(case['passed'] for case in report['cases'])
                    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
                    return report
                _, _, direct = ask('direct', 'Hello Maya! Tell me one short, interesting fact about the Moon.')
                save('direct', {'direct_route': direct['route'] == 'direct_answer', 'answer': bool(direct['answer']),
                                'no_tools': not direct['sdk_calls'] and not direct['exercise_calls']}, result=direct)
                _, _, health = ask('health', 'In general, what is cardiotoxicity after cancer treatment?')
                save('health_retrieval', {'answer_and_display_only_refs': answered(health),
                                         'retrieval': trace(health).get('retrieval_used') is True,
                                         'ontology_checked': bool(health['assessments']) and bool(health['assessments'][-1]['decision']['ontology_rule_checks'])}, result=health)

                question = 'Please interpret my latest blood pressure and explain whether it is safe given the cancer treatment I received.'
                clinical, config, clarification = ask('clinical', question)
                save('clarification', {'interrupt': bool(clarification['interrupts']),
                                      'scoped_lookup': clarification['sdk_calls'] == [{'resource': 'blood-pressure', 'subject': 'synthetic-patient'}],
                                      'observation_available': bool(clarification['observations']) and clarification['observations'][0]['status'] == 'ok',
                                      'rules_checked': bool(clarification['assessments']) and bool(clarification['assessments'][-1]['decision']['ontology_rule_checks'])}, result=clarification)
                if clarification['interrupts']:
                    resumed = turn(config, Command(resume='Doxorubicin, completed six months ago.'))
                    query = trace(resumed).get('retrieval_query', '')
                    save('clarification_resume', {'answer_and_display_only_refs': answered(resumed),
                         'no_refetch': not resumed['sdk_calls'],
                         'rewrite_preserves_context': 'doxorubicin' in query.lower() and '134' in query,
                         'memory_verified': 'doxorubicin' in json.dumps(workers[0].load_records(clinical.user_id, ['cancer_history'])).lower()}, result=resumed)
                else:
                    save('clarification_resume', {'preceding_interrupt': False})

                for name, prompt, tool in [
                    ('exercise_history', 'List my recorded Phyxio exercise sessions.', 'get_exercise_history'),
                    ('exercise_feedback', 'Review my latest recorded Phyxio exercise session and compare it with previous sessions.', 'get_latest_exercise_session')]:
                    _, _, result = ask(name, prompt)
                    save(name, {'route': result['route'] == 'phyxio_exercise_agent',
                                'tool': result['metadata'].get('executed_tool') == tool,
                                'history_read': any('getSessionList' in call for call in result['exercise_calls']),
                                'no_exercise_start': not any('run_routine' in call for call in result['exercise_calls']),
                                'answer': bool(result['answer'])}, result=result)

                _, _, biomarker = ask('readout', 'Show my latest blood pressure with its date, without interpreting it.')
                save('biomarker_readout', {'route': biomarker['route'] == 'biomarkers',
                    'public_scoped_call': biomarker['sdk_calls'] == [{'resource': 'blood-pressure', 'subject': 'synthetic-patient'}],
                    'values_units': all(value in biomarker['answer'] for value in ('134', '82', 'mmHg'))}, result=biomarker)
                _, _, urgent = ask('urgent', 'I have chest pain and shortness of breath right now. What should I do?')
                save('urgent', {'completed': urgent['route'] == 'health_rag' and not urgent['interrupts'] and bool(urgent['answer']),
                                'no_retrieval_delay': trace(urgent).get('retrieval_used') is False,
                                'no_lookup': not urgent['sdk_calls']}, result=urgent)

                if edges or medication:
                    medication_case()

                if edges:
                    _, _, mixed = ask('mixed', 'Bonjour Maya, please tell me a short fun fact about the Moon. Gracias!')
                    save('mixed_input', {'direct_answer': mixed['route'] == 'direct_answer' and bool(mixed['answer'])}, result=mixed)
                    omitted, _, legacy = ask('legacy', 'Hello Maya, tell me a short fun fact about the Moon.', omitted=True)
                    save('omitted_preference', {'configured_default': omitted.preferences.response_language == RESPONSE_LANGUAGES.default_response_language,
                                               'answer': bool(legacy['answer'])}, result=legacy, expected_language=omitted.preferences.response_language)
                    _, unknown_config, unknown = ask('unknown', 'My cardiac blood test was high. What does that mean?')
                    followup = turn(unknown_config, Command(resume="I don't know.")) if unknown['interrupts'] else unknown
                    save('unknown_followup', {'clarification': bool(unknown['interrupts']),
                        'no_guessed_lookup': not unknown['sdk_calls'] and not followup['sdk_calls'],
                        'completed': bool(followup['answer']) and not followup['interrupts']}, first=unknown, result=followup)

                    for name, empty in [('exercise_empty', True), ('exercise_unavailable', False)]:
                        reader = mirror.session_history_reader.return_value
                        previous = reader.getSessionList.side_effect
                        reader.getSessionList.side_effect = (lambda page: json.dumps({'page': page, 'sessions': []})) if empty else TimeoutError('Synthetic history timeout')
                        try:
                            _, _, result = ask(name, 'List my recorded Phyxio exercise sessions.')
                            save(name, {'tool': result['metadata'].get('executed_tool') == 'get_exercise_history',
                                        'answer': bool(result['answer']), 'no_fictional_history': '2026-10-12' not in result['answer']}, result=result)
                        finally:
                            reader.getSessionList.side_effect = previous

                    before = workers[0].load_records(clinical.user_id, ['cancer_history'])
                    changed_code = next(code for code in RESPONSE_LANGUAGES.languages if code not in (language, 'en'))
                    changed, _, recalled = ask('memory-switch', question, code=changed_code, user=clinical.user_id)
                    save('language_change_with_memory', {'same_participant': changed.user_id == clinical.user_id,
                        'existing_verified_memory': bool(before), 'answer_without_repeated_clarification': answered(recalled),
                        'saved_context_in_retrieval': any('doxorubicin' in (search['memory'] or '').lower() for search in recalled['searches']),
                        'medication_in_query': 'doxorubicin' in trace(recalled).get('retrieval_query', '').lower()}, result=recalled,
                        expected_language=changed_code, records_before=before)
                    _, _, friend = ask('other-person', "My friend takes bisoprolol. Is it okay with her cancer treatments?", user=clinical.user_id)
                    save('other_person', {'clarifies': bool(friend['interrupts']), 'no_patient_lookup': not friend['sdk_calls'],
                                         'other_subject': friend['subject'] == 'other',
                                         'no_borrowed_records': all('user_reported' not in (a['context'] or '') for a in friend['assessments'])}, result=friend)
                    _, _, isolated = ask('other-participant', question)
                    save('participant_isolation', {'needs_own_context': bool(isolated['interrupts'])}, result=isolated)

                    cache_question = 'In general, what is cardiotoxicity after cancer treatment?'
                    cache_user = f'language-{language}-cache'
                    _, _, first = ask('cache', cache_question, user=cache_user)
                    _, _, repeated = ask('cache', cache_question, user=cache_user)
                    _, _, changed = ask('cache', cache_question, user=cache_user, code=changed_code)
                    save('cache_language_scope', {'first_retrieves': trace(first).get('retrieval_used') is True,
                        'repeat_cache_matches_effective_context': bool(trace(first).get('memory_key')) and
                            trace(repeated).get('memory_hit') is (trace(first).get('memory_key') == trace(repeated).get('memory_key')),
                        'changed_language_retrieves': trace(changed).get('retrieval_used') is True,
                        'different_keys': trace(first).get('memory_key') != trace(changed).get('memory_key')},
                        first=first, repeated=repeated, result=changed, expected_language=changed_code)
                    for name, value in [('ambiguous_empty', []), ('unavailable', RuntimeError('Synthetic service failure'))]:
                        biomarkers.provider.client.get.side_effect = value if isinstance(value, Exception) else None
                        biomarkers.provider.client.get.return_value = value
                        _, _, result = ask(name, 'Show my latest blood pressure measurement, without interpreting it.')
                        save(name, {'unavailable': bool(result['observations']) and result['observations'][0]['status'] == 'unavailable',
                                    'no_values': '134' not in result['answer'], 'answer': bool(result['answer'])}, result=result)
                    biomarkers.provider.client.get.side_effect = None
                    biomarkers.provider.client.get.return_value = records(observation())
                    # Middleware hooks are bound when the graph is built. Inject the
                    # retriever at construction, as in the existing health harness.
                    with patch('maya.agents.health_rag.agent.build_vector_store') as vector:
                        vector.return_value.as_retriever.return_value = StubRetriever([])
                        graph = factory.build_supervisor(biomarker_service=biomarkers)
                    _, _, empty = ask('empty-evidence', 'In general, what is cardiotoxicity after cancer treatment?')
                    from maya.framework.conversation.responses import localize_messages
                    from maya.agents.health_rag.config import load_health_rag_config
                    expected_notice = localize_messages(load_health_rag_config().messages, language).no_evidence
                    save('missing_evidence_fixed_message', {'no_references': not empty['metadata'].get('presentation', {}).get('references'),
                         'configured_localized_fallback': empty['answer'] == expected_notice}, result=empty,
                         expected_notice=expected_notice)
        except Exception as error:
            response = getattr(error, 'response', None)
            headers = getattr(response, 'headers', {})
            save('unhandled_failure', {'completed': False}, error_type=type(error).__name__,
                 retry_headers={key: headers[key] for key in ('retry-after', 'retry-after-ms',
                    'x-ratelimit-remaining-tokens', 'x-ratelimit-remaining-requests') if key in headers})
        finally:
            for worker in workers:
                worker.close()
    report['passed'] = all(case['passed'] for case in report['cases'])
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('language')
    parser.add_argument('--edges', action='store_true')
    parser.add_argument('--medication', action='store_true')
    parser.add_argument('--only-medication', action='store_true')
    parser.add_argument('--label', default='', help='Retain a separate report for a focused rerun.')
    args = parser.parse_args()
    result = run(args.language, edges=args.edges, medication=args.medication, only_medication=args.only_medication, label=args.label)
    raise SystemExit(0 if result['passed'] else 1)
