from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets.custom_indic import CustomJsonlMCQDataset


bbh_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer'
)


bbh_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=(
                        '{question}\n'
                        'Answer:'
                    )
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


bbh_tasks = [
    'boolean_expressions',
    'causal_judgement',
    'date_understanding',
    'disambiguation_qa',
    'dyck_languages',
    'formal_fallacies',
    'geometric_shapes',
    'hyperbaton',
    'logical_deduction_five_objects',
    'logical_deduction_seven_objects',
    'logical_deduction_three_objects',
    'movie_recommendation',
    'multistep_arithmetic_two',
    'navigate',
    'object_counting',
    'penguins_in_a_table',
    'reasoning_about_colored_objects',
    'ruin_names',
    'salient_translation_error_detection',
    'snarks',
    'sports_understanding',
    'temporal_sequences',
    'tracking_shuffled_objects_five_objects',
    'tracking_shuffled_objects_seven_objects',
    'tracking_shuffled_objects_three_objects',
    'web_of_lies',
    'word_sorting',
]


bbh_datasets = []

for task in bbh_tasks:
    bbh_datasets.append(
        dict(
            abbr=f'bbh_{task}',
            type=CustomJsonlMCQDataset,
            path=f'/fsxnew/dhrumil.shah/opencompass_benchmarks/opencompass/data/bbh/{task}.jsonl',
            reader_cfg=bbh_reader_cfg,
            infer_cfg=bbh_infer_cfg,
            eval_cfg=dict(
                evaluator=dict(type=AccEvaluator),
                pred_role='BOT',
                pred_postprocessor=dict(type='bbh', task_name=task),
            ),
        )
    )