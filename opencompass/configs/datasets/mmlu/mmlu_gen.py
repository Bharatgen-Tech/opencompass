from mmengine.config import read_base

with read_base():
    from .mmlu_openai_simple_evals_gen_7efa2c import mmlu_datasets  # noqa: F401, F403