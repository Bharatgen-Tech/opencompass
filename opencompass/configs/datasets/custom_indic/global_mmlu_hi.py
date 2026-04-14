from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets.custom_indic import CustomJsonlMCQDataset
from opencompass.utils.text_postprocessors import match_answer_pattern


QUERY_TEMPLATE = """
नीचे दिए गए बहुविकल्पीय प्रश्न का उत्तर दें। अंतिम पंक्ति इस प्रारूप में होनी चाहिए:
'ANSWER: $LETTER' जहाँ LETTER A, B, C, या D में से एक है।

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()


global_mmlu_hi_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='answer'
)


global_mmlu_hi_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=QUERY_TEMPLATE),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


global_mmlu_hi_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(
        type=match_answer_pattern,
        answer_pattern=r'(?i)ANSWER\s*:\s*([A-D])'
    )
)


global_mmlu_hi_datasets = [
    dict(
        abbr='global_mmlu_hi_dev',
        type=CustomJsonlMCQDataset,
        path='/fsxnew/dhrumil.shah/opencompass_benchmarks/data/global_mmlu_hi/dev.jsonl',
        question_key='question',
        options_keys=['A', 'B', 'C', 'D'],
        answer_key='answer',
        reader_cfg=global_mmlu_hi_reader_cfg,
        infer_cfg=global_mmlu_hi_infer_cfg,
        eval_cfg=global_mmlu_hi_eval_cfg,
    ),
    dict(
        abbr='global_mmlu_hi_test',
        type=CustomJsonlMCQDataset,
        path='/fsxnew/dhrumil.shah/opencompass_benchmarks/data/global_mmlu_hi/test.jsonl',
        question_key='question',
        options_keys=['A', 'B', 'C', 'D'],
        answer_key='answer',
        reader_cfg=global_mmlu_hi_reader_cfg,
        infer_cfg=global_mmlu_hi_infer_cfg,
        eval_cfg=global_mmlu_hi_eval_cfg,
    )
]