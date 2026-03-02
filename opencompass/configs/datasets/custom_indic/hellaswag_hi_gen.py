from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer 
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HellaswagDataset 

# ✅ Standard HellaSwag Reader
hellaswag_hi_reader_cfg = dict(
    input_columns=['ctx', 'endings'],
    output_column='label'
)

# ✅ Hindi-Specific Generative Prompt
hellaswag_hi_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='अधूरे वाक्य को पूरा करने के लिए सबसे उपयुक्त विकल्प चुनें:\nविवरण: {ctx}\n\nविकल्प:\nA. {endings[0]}\nB. {endings[1]}\nC. {endings[2]}\nD. {endings[3]}\n\nउत्तर: '),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

hellaswag_hi_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
)

hellaswag_hi_datasets = [
    dict(
        abbr='hellaswag_hi',
        type=HellaswagDataset,
        # ✅ Pointing to your exact /fsx/ path
        path='/fsxnew/dhrumil.shah/opencompass_benchmarks/data/hellaswag_hi',
        reader_cfg=hellaswag_hi_reader_cfg,
        infer_cfg=hellaswag_hi_infer_cfg,
        eval_cfg=hellaswag_hi_eval_cfg,
    )
]