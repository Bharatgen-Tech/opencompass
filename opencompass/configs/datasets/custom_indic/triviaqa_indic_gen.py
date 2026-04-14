from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets.custom_indic import CustomArrowMCQDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

langs = ['hi', 'bn', 'gu', 'kn', 'ml', 'mr', 'or', 'pa', 'ta', 'te']

triviaqa_indic_mcq_datasets = []

for lang in langs:
    triviaqa_indic_mcq_datasets.append(
        dict(
            abbr=f'triviaqa_indic_{lang}',
            type=CustomArrowMCQDataset,
            path=f'/fsxnew/dhrumil.shah/opencompass_benchmarks/data/trivia_qa_indic_mcq/{lang}',
            split='validation',
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
                            ),
                        ],
                    ),
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