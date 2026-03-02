from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets.custom_indic import CustomJsonlMCQDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

sanskriti_tasks = ['association', 'country_prediction', 'general_awareness', 'state_prediction']
sanskriti_datasets = []

for task in sanskriti_tasks:
    sanskriti_datasets.append(
        dict(
            abbr=f'sanskriti_{task}',
            type=CustomJsonlMCQDataset,
            path=f'/fsxnew/dhrumil.shah/opencompass_benchmarks/data/sanskriti/{task}_questions.jsonl',
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
                            dict(role='HUMAN', prompt='Question: {question}\nAnswer:'),
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