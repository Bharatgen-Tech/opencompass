from opencompass.openicl.icl_evaluator import AveragePPLEvaluator
from opencompass.datasets.custom_indic import CustomJsonlPPLDataset

pile_10k_datasets = [
    dict(
        abbr='pile_10k',
        type=CustomJsonlPPLDataset,
        path='/fsxnew/dhrumil.shah/opencompass_benchmarks/opencompass/data/pile_10k/test.jsonl',
        text_key='text',

        reader_cfg=dict(
            input_columns=['text'],
            output_column='text'
        ),

        # ❗ NO prompt_template
        # ❗ NO retriever
        # ❗ NO PPLInferencer

        infer_cfg=dict(
            inferencer=dict(type='LMInferencer')  # <-- key fix
        ),

        eval_cfg=dict(
            evaluator=dict(type=AveragePPLEvaluator)
        )
    )
]
