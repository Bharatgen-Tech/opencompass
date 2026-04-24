from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset
from opencompass.utils.text_postprocessors import first_option_postprocess


# ✅ Convert HF label (0–3) → A/B/C/D
def label_to_abcd(label):
    return ['A', 'B', 'C', 'D'][label]


# ✅ Reader: map HF schema → OpenCompass
hellaswag_hi_reader_cfg = dict(
    input_columns=['ctx', 'endings'],
    output_column='label',
    output_postprocess=label_to_abcd,
)


# ✅ Prompt (uses endings list correctly)
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
                        'A. {endings[0]}\n'
                        'B. {endings[1]}\n'
                        'C. {endings[2]}\n'
                        'D. {endings[3]}\n\n'
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


# ✅ Evaluation (unchanged, now compatible)
hellaswag_hi_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)


# ✅ Dataset (FINAL FIX)
hellaswag_hi_datasets = [
    dict(
        abbr='hellaswag_hi',
        type=HFDataset,  # ✅ correct loader for HF
        path='ai4bharat/hellaswag-translated',
        name='hi',       # ✅ correct config (you verified this)
        split='validation',  # ✅ THIS is the correct key (NOT test_split)
        reader_cfg=hellaswag_hi_reader_cfg,
        infer_cfg=hellaswag_hi_infer_cfg,
        eval_cfg=hellaswag_hi_eval_cfg,
    )
]