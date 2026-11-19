"""Legacy deterministic model doubles for workflow tests.

Privacy boundary tests exercise real LangChain agents separately.
"""


def model_call_stub(model, schema=None, **kwargs):
    return model.with_structured_output(schema, **kwargs) if schema is not None else model
