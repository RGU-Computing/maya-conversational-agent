# OntoModel Translation Review

These models are draft translations from screenshots in `maya/models_screenshots`.
They should be reviewed by the ontology owner before runtime use.

## Runtime Status

Runtime prompt context is generated from selected models, but procedure contracts
marked `candidate_strict` are not yet deterministic graph gates. Promote them only
after the ontology owner confirms their triggers, required inputs, order, and
failure/clarification behavior.

Only these reviewed memory bindings currently accept writes:

- `clinical_profile.medication_records`
- `clinical_profile.condition_records`
- `cancer_history.treatment_records`

All other memory bindings remain draft and are ignored by the memory writer.

## Verification

Deterministic contract, persistence, retry, correction, dependency, and model
evolution tests run with:

```bash
python -m unittest discover -s maya/tests -p 'test_*.py' -v
```

Semantic save/no-save behavior must be tested against the configured Azure model,
not a lexical Python substitute:

```bash
python -m maya.tests.live_ontology_memory_scenarios
python -m maya.tests.live_supervisor_routing_scenarios
```

The live matrix is intentionally excluded from normal unit discovery because it
requires Azure credentials, outbound network access, time, and model usage.
It covers exact extracted properties as well as routing, multilingual input,
corrections, mixed fact/questions, hypotheticals, uncertainty, third-party and
quoted claims, and prompt-injection attempts.

## Review Priorities

1. Confirm relation directions for arrows that cross or curve between distant nodes.
2. Confirm whether grouped boxes labelled "subclasses OR instances" should be
   treated as true subclasses, controlled value sets, or example instances.
3. Confirm labels that appear semantically broader or narrower than their target.
4. Confirm which procedure candidates are strict requirements versus soft guidance.
5. Confirm which models are global knowledge models and which are user-memory models.

## Known Uncertainties

- `clinical_profile`: `ClinicalProfile -> ClinicalRecord` is labelled
  `includesConditionRecord` in the screenshot, but the target is `ClinicalRecord`.
  It may need to become `includesClinicalRecord`.
- `cancer_history`: `ChestRadiotherapyExposure` is close to the image edge; the
  label is readable, but the ontology owner should confirm it.
- `exercise`: `PerformedExerciseSession realises PlannedExerciseSession` is the
  interpreted direction from the visual arrow. Confirm this direction.
- `intervention_careplan`: The arrow from `Intervention` to
  `CarePlanActivity` appears unlabeled. It is captured as an `unlabeled`
  low-confidence relation.
- `intervention_careplan`: `CarePlanParticipation` appears to be an n-ary
  participation record linking `CarePlan`, `Person`, and `PersonRole`.
- `measurement_biomarker`: `MeasurementContext` and `DataSource` are explicitly
  marked as "subclasses OR instances"; keep this flexible until confirmed.
- `knowledge_resource`: This may be a global provenance model more than a
  user-memory model. User-specific memory should probably store derivations,
  displayed recommendations, and reviews rather than whole guideline resources.

## Future Update Workflow

When an ontology screenshot changes:

1. Add or replace the screenshot in `maya/models_screenshots`.
2. Update the matching `model.json` version.
3. Keep existing stable IDs unless the concept meaning truly changed.
4. Add aliases for renamed labels.
5. Mark removed concepts as deprecated before deleting them.
6. Update `memory_bindings` only after deciding migration behavior.
7. Update `procedure_contracts` separately from structural nodes/relations.
8. Re-run JSON and reference validation.

This keeps future memory migration clean: labels can change often, while memory
paths and stable IDs remain controlled.
