from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HellaswagDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

hellaswag_hi_reader_cfg = dict(
    input_columns=['ctx', 'A', 'B', 'C', 'D'],
    output_column='label'
)

hellaswag_hi_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=(
                        'अधूरे वाक्य को पूरा करने के लिए सबसे उपयुक्त विकल्प चुनें:\n'
                        'विवरण: {ctx}\n\n'
                        'विकल्प:\n'
                        'A. {A}\n'
                        'B. {B}\n'
                        'C. {C}\n'
                        'D. {D}\n\n'
                        'केवल A, B, C, या D में उत्तर दें।\n'
                        'उत्तर:'
                    )
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

hellaswag_hi_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)

hellaswag_hi_datasets = [
    dict(
        abbr='hellaswag_hi',
        type=HellaswagDataset,
        path='/fsxnew/dhrumil.shah/opencompass_benchmarks/opencompass/data/hellaswag_indic/hi_label_fixed.jsonl',
        reader_cfg=hellaswag_hi_reader_cfg,
        infer_cfg=hellaswag_hi_infer_cfg,
        eval_cfg=hellaswag_hi_eval_cfg,
    )
]