from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset

# Mapping for Indic languages in the SarvamAI dataset
langs = ['hi', 'bn', 'gu', 'kn', 'ml', 'mr', 'or', 'pa', 'ta', 'te']

triviaqa_indic_mcq_datasets = []

for lang in langs:
    triviaqa_indic_mcq_datasets.append(
        dict(
            abbr=f'triviaqa_indic_{lang}',
            type=HFDataset,
            path='/fsxnew/dhrumil.shah/opencompass_benchmarks/data/trivia_qa_indic_mcq/{lang}',
            name=lang,
            reader_cfg=dict(
                input_columns=['question', 'options'],
                output_column='answer'
            ),
            infer_cfg=dict(
                prompt_template=dict(
                    type=PromptTemplate,
                    template=dict(
                        round=[
                            dict(role='HUMAN', prompt='प्रश्न: {question}\nविकल्प: {options}\nउत्तर:'),
                        ],
                    ),
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