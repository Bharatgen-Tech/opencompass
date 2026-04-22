from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets.custom_indic import CustomJsonlMCQDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

langs = [
    'bengali', 'english', 'gujarati', 'hindi', 'kannada',
    'malayalam', 'marathi', 'odia', 'punjabi', 'tamil', 'telugu'
]

milu_datasets = []

for lang in langs:
    milu_datasets.append(
        dict(
            abbr=f'milu_{lang}',
            type=CustomJsonlMCQDataset,
            path=f'/fsxnew/dhrumil.shah/opencompass_benchmarks/opencompass/data/milu/{lang}.jsonl',
            question_key='question',
            options_keys=['A', 'B', 'C', 'D'],
            answer_key='answer',
            reader_cfg=dict(
                input_columns=['question', 'A', 'B', 'C', 'D'],
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
                                    'Question: {question}\n'
                                    'A. {A}\n'
                                    'B. {B}\n'
                                    'C. {C}\n'
                                    'D. {D}\n'
                                    'Answer with only A, B, C, or D.\n'
                                    'Answer:'
                                )
                            )
                        ]
                    )
                ),
                retriever=dict(type=ZeroRetriever),
                inferencer=dict(type=GenInferencer),
            ),
            eval_cfg=dict(
                evaluator=dict(type=AccEvaluator),
                pred_role='BOT',
                pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
            )
        )
    )