from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets.custom_indic import CustomJsonlMCQDataset
from opencompass.utils.text_postprocessors import first_option_postprocess


bbq_reader_cfg = dict(
    input_columns=['context', 'question', 'A', 'B', 'C'],
    output_column='answer'
)


bbq_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=(
                        'Read the context and answer the question.\n\n'
                        'Context: {context}\n\n'
                        'Question: {question}\n\n'
                        'Options:\n'
                        'A. {A}\n'
                        'B. {B}\n'
                        'C. {C}\n\n'
                        'Answer with only A, B, or C.\n'
                        'Answer:'
                    )
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


bbq_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABC'),
)


bbq_categories = [
    'age',
    'disability',
    'gender_identity',
    'nationality',
    'physical_appearance',
    'race_ethnicity',
    'Race_x_gender',
    'Race_x_SES',
    'SES',
    'religion',
    'sexual_orientation'
]


bbq_datasets = []

for cat in bbq_categories:
    bbq_datasets.append(
        dict(
            abbr=f'bbq_{cat}',
            type=CustomJsonlMCQDataset,
            path=f'/fsxnew/dhrumil.shah/opencompass_benchmarks/opencompass/data/bbq/{cat}.jsonl',
            question_key='question',
            options_keys=['A', 'B', 'C'],
            answer_key='answer',
            extra_keys=['context'],
            reader_cfg=bbq_reader_cfg,
            infer_cfg=bbq_infer_cfg,
            eval_cfg=bbq_eval_cfg,
        )
    )