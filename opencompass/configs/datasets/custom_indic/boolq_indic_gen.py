from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets.custom_indic import CustomJsonlMCQDataset

BASE_PATH = '/fsxnew/dhrumil.shah/opencompass_benchmarks/opencompass/data/boolq_indic'

languages = [
    'en',
    'hi',
    'bn',
    'mr',
    'pa',
    'gu',
    'kn',
    'ml',
    'ta',
    'te',
    'or'
]

boolq_indic_datasets = []

for lang in languages:

    boolq_indic_datasets.append(
        dict(
            abbr=f'boolq_indic_{lang}',

            type=CustomJsonlMCQDataset,

            path=f'{BASE_PATH}/{lang}.jsonl',

            question_key='question',

            context_key='passage',

            answer_key='answer',

            reader_cfg=dict(
                input_columns=['passage', 'question'],
                output_column='answer'
            ),

            infer_cfg=dict(
                prompt_template=dict(
                    type=PromptTemplate,
                    template=dict(
                        round=[
                            dict(
                                role='HUMAN',
                                prompt=(
                                    'Passage: {passage}\n\n'
                                    'Question: {question}\n\n'
                                    "Answer with only 'yes' or 'no'.\n"
                                    'Answer:'
                                )
                            ),
                        ]
                    )
                ),

                retriever=dict(type=ZeroRetriever),

                inferencer=dict(type=GenInferencer),
            ),

            eval_cfg=dict(
                evaluator=dict(type=AccEvaluator),

                pred_role='BOT',
            )
        )
    )