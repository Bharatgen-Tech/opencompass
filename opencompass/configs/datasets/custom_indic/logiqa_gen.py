from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets.custom_indic import CustomJsonlMCQDataset
from opencompass.utils.text_postprocessors import first_option_postprocess


logiqa_reader_cfg = dict(
    input_columns=['context', 'question', 'A', 'B', 'C', 'D'],
    output_column='answer'
)


logiqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=(
                        'Read the passage and answer the question.\n\n'
                        'Passage: {context}\n\n'
                        'Question: {question}\n\n'
                        'Options:\n'
                        'A. {A}\n'
                        'B. {B}\n'
                        'C. {C}\n'
                        'D. {D}\n\n'
                        'Answer with only A, B, C, or D.\n'
                        'Answer:'
                    )
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


logiqa_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)


logiqa_datasets = [
    dict(
        abbr='logiqa_dev',
        type=CustomJsonlMCQDataset,
        path='/fsxnew/dhrumil.shah/opencompass_benchmarks/data/logiqa/dev.jsonl',
        question_key='question',
        options_keys=['A', 'B', 'C', 'D'],
        answer_key='answer',
        extra_keys=['context'],
        reader_cfg=logiqa_reader_cfg,
        infer_cfg=logiqa_infer_cfg,
        eval_cfg=logiqa_eval_cfg,
    ),
    dict(
        abbr='logiqa_test',
        type=CustomJsonlMCQDataset,
        path='/fsxnew/dhrumil.shah/opencompass_benchmarks/data/logiqa/test.jsonl',
        question_key='question',
        options_keys=['A', 'B', 'C', 'D'],
        answer_key='answer',
        extra_keys=['context'],
        reader_cfg=logiqa_reader_cfg,
        infer_cfg=logiqa_infer_cfg,
        eval_cfg=logiqa_eval_cfg,
    ),
]