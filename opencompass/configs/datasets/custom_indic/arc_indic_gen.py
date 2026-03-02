from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets.custom_indic import CustomJsonlMCQDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

# The 10 languages you have in /fsx/opencompass_data/data/arc_indic/
arc_langs = ['bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'or', 'pa', 'ta', 'te']
arc_indic_datasets = []

for lang in arc_langs:
    arc_indic_datasets.append(
        dict(
            abbr=f'arc_indic_{lang}',
            type=CustomJsonlMCQDataset,
            path=f'/fsxnew/dhrumil.shah/opencompass_benchmarks/data/arc_indic/{lang}.jsonl',
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
                            dict(role='HUMAN', prompt='Question: {question}\nAnswer:')
                        ]
                    )
                ),
                retriever=dict(type=ZeroRetriever),
                inferencer=dict(type=PPLInferencer),
            ),
            eval_cfg=dict(
                evaluator=dict(type=AccEvaluator),
                pred_role='BOT',
            )
        )
    )