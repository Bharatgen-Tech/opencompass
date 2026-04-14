from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets.custom_indic import CustomJsonlMCQDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

import os

base_path = "/fsxnew/dhrumil.shah/opencompass_benchmarks/data/indic_param_indic"

indic_param_indic_datasets = []


for file in sorted(os.listdir(base_path)):
    if not file.endswith(".jsonl"):
        continue

    lang = file.replace(".jsonl", "")

    indic_param_indic_datasets.append(
        dict(
            abbr=f'indic_param_indic_{lang}',
            type=CustomJsonlMCQDataset,

            path=os.path.join(base_path, file),

            # 🔥 IMPORTANT (THIS replaces your custom class)
            question_key='question',
            options_keys=['A', 'B', 'C', 'D'],
            answer_key='answer',

            reader_cfg=dict(
                input_columns=['question', 'A', 'B', 'C', 'D', 'lang'],
                output_column='answer',
            ),

            infer_cfg=dict(
                prompt_template=dict(
                    type=PromptTemplate,
                    template=dict(
                        round=[
                            dict(
                                role='HUMAN',
                                prompt=(
                                    "Question: {question}\n"
                                    "Options:\n"
                                    "A) {A}\n"
                                    "B) {B}\n"
                                    "C) {C}\n"
                                    "D) {D}\n"
                                    "The above question is written in {lang} language. Please analyze "
                                    "the question and options carefully, and select the correct answer.\n"
                                    "Respond ONLY with one letter (A, B, C, or D) corresponding to the "
                                    "correct option. Do not provide any explanation or additional text."
                                )
                            )
                        ]
                    )
                ),
                retriever=dict(type=ZeroRetriever),

                inferencer=dict(
                    type=GenInferencer,
                ),
            ),

            eval_cfg=dict(
                evaluator=dict(type=AccEvaluator),
                pred_role='BOT',
                pred_postprocessor=dict(
                    type=first_option_postprocess,
                    options='ABCD'
                ),
            )
        )
    )


