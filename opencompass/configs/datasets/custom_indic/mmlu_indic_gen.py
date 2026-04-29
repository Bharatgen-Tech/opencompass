from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import MMLUIndicDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

# ── Reader ────────────────────────────────────────────────────────────────────
mmlu_indic_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
)

# ── Prompt ────────────────────────────────────────────────────────────────────
mmlu_indic_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt = (
                        'Choose the correct answer to the following question:\n'
                        'Question: {input}\n\n'
                        'Options:\n'
                        'A. {A}\n'
                        'B. {B}\n'
                        'C. {C}\n'
                        'D. {D}\n\n'
                        'Respond with only A, B, C, or D.\n'
                        'Answer:'
                    ),
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# ── Evaluator ─────────────────────────────────────────────────────────────────
mmlu_indic_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)

# ── Dataset list — one entry per language ─────────────────────────────────────
_LANGUAGES = [
    'as_in', 'bn_in', 'bodo_in', 'dogri_in', 'gu_in',
    'hi_in', 'kashmiri_in', 'kn_in', 'konkani_in', 'mai_in',
    'manipuri_bng_in', 'ml_in', 'mr_in', 'ne_in', 'or_in',
    'pa_in', 'sa_in', 'santali_in', 'sdd_in', 'ta_in', 'te_in',
]

mmlu_indic_datasets = [
    dict(
        abbr=f'mmlu_indic_{lang}',
        type=MMLUIndicDataset,
        path='/fsxnew/dhrumil.shah/opencompass_benchmarks/opencompass/data/mmlu_indic',   # relative to opencompass root
        name=lang,                            # resolves to mmlu_{lang}.jsonl
        reader_cfg=mmlu_indic_reader_cfg,
        infer_cfg=mmlu_indic_infer_cfg,
        eval_cfg=mmlu_indic_eval_cfg,
    )
    for lang in _LANGUAGES
]