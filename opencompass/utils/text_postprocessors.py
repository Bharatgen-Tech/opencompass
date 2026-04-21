import re
from typing import Callable, Optional, Union

from opencompass.registry import TEXT_POSTPROCESSORS


@TEXT_POSTPROCESSORS.register_module('general')
def general_postprocess(text: str) -> str:
    # Cut off the first newline, period, or comma
    truncated_text = re.split(r'[\n.,]', text, 1)[0]

    # Remove punctuation
    no_punctuation = re.sub(r'[^\w\s]', '', truncated_text)

    # Remove article
    no_articles = re.sub(r'\b(a|an|the)\b',
                         '',
                         no_punctuation,
                         flags=re.IGNORECASE)

    # Remove duplicated blank spaces
    cleaned_text = re.sub(r'\s+', ' ', no_articles).strip()

    return cleaned_text


@TEXT_POSTPROCESSORS.register_module('general_cn')
def general_cn_postprocess(text: str) -> str:
    truncated_text = re.split(r'[\n.,]', text, 1)[0]

    no_punctuation = re.sub(r'[^\w\s]', '', truncated_text)

    no_articles = re.sub(r'\b(a|an|the)\b',
                         '',
                         no_punctuation,
                         flags=re.IGNORECASE)

    cleaned_text = re.sub(r'\s+', ' ', no_articles).strip()
    import jieba

    cleaned_text = ' '.join(jieba.cut(text))
    return cleaned_text


@TEXT_POSTPROCESSORS.register_module('first-capital')
def first_capital_postprocess(text: str) -> str:
    for t in text:
        if t.isupper():
            return t
    return ''


@TEXT_POSTPROCESSORS.register_module('last-capital')
def last_capital_postprocess(text: str) -> str:
    for t in text[::-1]:
        if t.isupper():
            return t
    return ''


@TEXT_POSTPROCESSORS.register_module('think_pred')
def think_pred_postprocess(
    prediction: str,
    re_pattern: str,
) -> str:
    match = re.search(re_pattern, prediction)
    if match:
        return match.group(1).strip()
    else:
        return prediction


def first_option_postprocess(text: str, options: str, cushion=True) -> str:
    """Find first valid option for text."""

    # yapf: disable
    # flake8: noqa: W605
    patterns = [
        f'答案是?\s*([{options}])',
        f'答案是?\s*：\s*([{options}])',
        f'答案是?\s*:\s*([{options}])',
        f'答案选项应?该?是\s*([{options}])',
        f'答案选项应?该?为\s*([{options}])',
        f'答案应该?是\s*([{options}])',
        f'答案应该?选\s*([{options}])',
        f'答案选项为?\s*：\s*([{options}])',
        f'答案选项为?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        f'答案选项是?\s*:\s*([{options}])',
        f'答案为\s*([{options}])',
        f'答案选\s*([{options}])',
        f'选择?\s*([{options}])',
        f'故选?\s*([{options}])'
        f'只有选?项?\s?([{options}])\s?是?对',
        f'只有选?项?\s?([{options}])\s?是?错',
        f'只有选?项?\s?([{options}])\s?不?正确',
        f'只有选?项?\s?([{options}])\s?错误',
        f'说法不?对选?项?的?是\s?([{options}])',
        f'说法不?正确选?项?的?是\s?([{options}])',
        f'说法错误选?项?的?是\s?([{options}])',
        f'([{options}])\s?是正确的',
        f'([{options}])\s?是正确答案',
        f'选项\s?([{options}])\s?正确',
        f'所以答\s?([{options}])',
        f'所以\s?([{options}][.。$]?$)',
        f'所有\s?([{options}][.。$]?$)',
        f'[\s，：:,]([{options}])[。，,\.]?$',
        f'[\s，,：:][故即]([{options}])[。\.]?$',
        f'[\s，,：:]因此([{options}])[。\.]?$',
        f'[是为。]\s?([{options}])[。\.]?$',
        f'因此\s?([{options}])[。\.]?$',
        f'显然\s?([{options}])[。\.]?$',
        f'答案是\s?(\S+)(?:。|$)',
        f'答案应该是\s?(\S+)(?:。|$)',
        f'答案为\s?(\S+)(?:。|$)',
        f'(?i)ANSWER\s*:\s*([{options}])',
        f'[Tt]he answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he answer is:?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        f'[Tt]he answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct answer option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he answer to the question is:?\s+\(?([{options}])\)?',
        f'^选项\s?([{options}])',
        f'^([{options}])\s?选?项',
        f'(\s|^)[{options}][\s。，,：:\.$]',
        f'1.\s?(.*?)$',
        f'1.\s?([{options}])[.。$]?$',
    ]
    cushion_patterns = [
        f'([{options}]):',
        f'([{options}])',
    ]
    # flake8: noqa
    # yapf: enable

    if cushion:
        patterns.extend(cushion_patterns)
    for pattern in patterns:
        text = text.strip()
        match = re.search(pattern, text, re.DOTALL)
        if match:
            if match.group(1) is not None and match.group(1) != '':
                outputs = match.group(1)
            else:
                outputs = match.group(0)
            for i in options:
                if i in outputs:
                    return i
    return ''


@TEXT_POSTPROCESSORS.register_module('first-capital-multi')
def first_capital_postprocess_multi(text: str) -> str:
    match = re.search(r'([A-D]+)', text)
    if match:
        return match.group(1)
    return ''


def last_option_postprocess(text: str, options: str) -> str:
    match = re.findall(rf'([{options}])', text)
    if match:
        return match[-1]
    return ''


def first_number_postprocess(text: str) -> float:
    """Return the first number in a string."""
    # regex pattern to match numbers (both integers and decimals)
    pattern = r'(-?\d*\.?\d+)'

    # search the string for the pattern
    match = re.search(pattern, text)

    # if a match is found, return it. Otherwise, return None.
    return float(match.group(1)) if match else None


@TEXT_POSTPROCESSORS.register_module('multiple-select')
def multiple_select_postprocess(text: str) -> str:
    ret = set([t for t in text if t.isupper()])
    return ''.join(sorted(ret))


@TEXT_POSTPROCESSORS.register_module('specific-xml-tag')
def xml_tag_postprocessor(text, tag):
    """Extracts content enclosed within a specified XML-style tag from a
    string.

    Args:
        texts: The input string containing XML-style tags.
        tag: The XML-style tag to extract content from (e.g., "<conclude>").  Must include the angle brackets.

    Returns:
        The content enclosed within the specified tag, or None if the tag is not found.
    """

    # Use a regular expression to find the content within the specified tag.  This handles cases where the tag might appear multiple times.
    matches = re.findall(
        rf'{tag}(.*?)</{tag[1:-1]}>', text,
        re.DOTALL)  # re.DOTALL allows . to match newline characters

    if matches:
        # Only keep the last one
        output = matches[-1].strip(
        )  # Extract the content and remove leading/trailing whitespace
    else:
        output = 'NO ANSWER FOUND'

    return output


def general_eval_wrapper_postprocess(text: str,
                                     postprocess: Optional[Union[
                                         str, Callable]] = None,
                                     **kwargs) -> str:
    """Wrapper for eval text repr. Especially for chatglmpro.

    Args:
        text(str): Text to be postprocessed.
        postprocess(Callable, optional): Original post processing function.
            Defaults to None.
        **kwargs: Other necessary kwargs for post processing function.
    """
    try:
        text = eval(text)
    except Exception:
        # in case empty input or other error, skip eval
        pass

    if postprocess:
        if isinstance(postprocess, str):
            postprocess = TEXT_POSTPROCESSORS.get(postprocess)
        return postprocess(text, **kwargs)
    else:
        return text


@TEXT_POSTPROCESSORS.register_module()
def match_answer_pattern(response_text: str, answer_pattern: str):
    match = re.search(answer_pattern, response_text)
    extracted_answer = match.group(1) if match else ''
    return extracted_answer


@TEXT_POSTPROCESSORS.register_module('extract-non-reasoning-content')
def extract_non_reasoning_content(
    text: str,
    think_start_token: str = '<think>',
    think_end_token: str = '</think>',
) -> str:
    """Extract content after the last reasoning tag from text.

    When only end token is present, returns content after the end token.
    When both tokens are present, removes all content between start and end tokens.

    Args:
        text (str): Input text containing reasoning tags.
        think_start_token (str, optional): Start token for reasoning section. Defaults to '<think>'.
        think_end_token (str, optional): End token for reasoning section. Defaults to '</think>'.

    Returns:
        str: Processed text after removing reasoning sections.

    Examples:
        >>> # When only end token exists
        >>> text = "This is a test.</think> How are you?"
        >>> extract_non_reasoning_content(text)
        'How are you?'

        >>> # When both tokens exist
        >>> text = "Start<think>reasoning here</think> End"
        >>> extract_non_reasoning_content(text)
        'Start End'
    """
    # If text contains only end token, split by end token and take the last part
    if think_start_token not in text and think_end_token in text:
        return text.split(think_end_token)[-1].strip()

    # Original behavior for complete tag pairs
    reasoning_regex = re.compile(rf'{think_start_token}(.*?){think_end_token}',
                                 re.DOTALL)
    non_reasoning_content = reasoning_regex.sub('', text).strip()
    return non_reasoning_content


import unicodedata
import re

@TEXT_POSTPROCESSORS.register_module('indic_mcq')
def indic_mcq_postprocess(text: str, options: str = 'ABCD') -> str:
    """
    Robust MCQ answer extractor for Indic + English LLM outputs.
    Handles: Latin letters, Devanagari, Tamil, Telugu, Bengali,
             Kannada, Malayalam, Gujarati, Punjabi, Odia, ordinals.
    """
    if not text:
        return ''

    # 1. Normalize unicode (NFC) — handles diacritic encoding variants
    text = unicodedata.normalize('NFC', text.strip())
    upper = text.upper()

    # 2. Direct single-char Latin match (most common case)
    if upper in options:
        return upper

    # 3. Explicit "Answer: X" / "Option X" / "(X)" patterns
    patterns = [
        r'\bAnswer\s*[:\-]?\s*([A-D])\b',
        r'\bOption\s+([A-D])\b',
        r'^\s*\(?([A-D])\)?[\s\.\)]',   # leading (A) or A. or A)
        r'([A-D])\s*(?:is correct|is the answer|\.?\s*$)',
    ]
    for pat in patterns:
        m = re.search(pat, upper, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()

    # 4. Script-specific letter mappings (per-language)
    # Each script has its own letter ordering that maps to A/B/C/D
    SCRIPT_MAPS = {
        # Devanagari (Hindi, Marathi, Sanskrit)
        'अ': 'A', 'ब': 'B', 'स': 'C', 'ड': 'D',
        'बी': 'B', 'सी': 'C', 'डी': 'D',
        # Full option labels common in Hindi MCQs
        'विकल्प अ': 'A', 'विकल्प ब': 'B', 'विकल्प स': 'C', 'विकल्प ड': 'D',

        # Bengali
        'ক': 'A', 'খ': 'B', 'গ': 'C', 'ঘ': 'D',

        # Tamil
        'அ': 'A', 'ஆ': 'B', 'இ': 'C', 'ஈ': 'D',

        # Telugu
        'అ': 'A', 'బ': 'B', 'స': 'C', 'డ': 'D',

        # Kannada
        'ಅ': 'A', 'ಬ': 'B', 'ಸ': 'C', 'ಡ': 'D',

        # Malayalam
        'എ': 'A', 'ബി': 'B', 'സി': 'C', 'ഡി': 'D',

        # Gujarati
        'અ': 'A', 'બ': 'B', 'ક': 'C', 'ડ': 'D',

        # Punjabi (Gurmukhi)
        'ਏ': 'A', 'ਬੀ': 'B', 'ਸੀ': 'C', 'ਡੀ': 'D',

        # Odia
        'ଅ': 'A', 'ବ': 'B', 'ସ': 'C', 'ଡ': 'D',

        # Urdu (Arabic script)
        'الف': 'A', 'ب': 'B', 'ج': 'C', 'د': 'D',
    }

    # Exact match first (avoids substring collisions)
    if text in SCRIPT_MAPS:
        return SCRIPT_MAPS[text]

    # 5. Ordinal word forms (models sometimes say "first option", "पहला", etc.)
    ORDINALS = {
        # English
        'FIRST': 'A', 'SECOND': 'B', 'THIRD': 'C', 'FOURTH': 'D',
        'ONE': 'A', 'TWO': 'B', 'THREE': 'C', 'FOUR': 'D',
        '1ST': 'A', '2ND': 'B', '3RD': 'C', '4TH': 'D',
        '1': 'A', '2': 'B', '3': 'C', '4': 'D',

        # Hindi ordinals
        'पहला': 'A', 'पहली': 'A', 'दूसरा': 'B', 'दूसरी': 'B',
        'तीसरा': 'C', 'तीसरी': 'C', 'चौथा': 'D', 'चौथी': 'D',
    }
    if upper in ORDINALS:
        return ORDINALS[upper]
    for word, val in ORDINALS.items():
        if word in upper:
            return val

    # 6. Substring search for script maps (last resort, ordered longest-first
    #    to avoid e.g. 'ब' matching inside 'बी')
    for key in sorted(SCRIPT_MAPS, key=len, reverse=True):
        if key in text:
            return SCRIPT_MAPS[key]

    # 7. Last resort: find any A/B/C/D in the string
    m = re.search(r'\b([A-D])\b', upper)
    if m:
        return m.group(1)

    return ''





# # ──────────────────────────────────────────────────────────────────────────────
# # BBH (BIG-Bench Hard) postprocessor
# # ──────────────────────────────────────────────────────────────────────────────

# # Tasks grouped by their answer type
# _BBH_YESNO_TASKS = {
#     'causal_judgement',
#     'navigate',
#     'sports_understanding',
#     'web_of_lies',
# }

# _BBH_TRUEFALSE_TASKS = {
#     'boolean_expressions',
# }

# _BBH_VALID_INVALID_TASKS = {
#     'formal_fallacies',
# }

# _BBH_NUMERIC_TASKS = {
#     'multistep_arithmetic_two',
#     'object_counting',
# }

# _BBH_FREETEXT_TASKS = {
#     'dyck_languages',
#     'word_sorting',
# }

# # All remaining tasks are multiple choice (A-K)
# _BBH_MCQ_TASKS = {
#     'date_understanding',
#     'disambiguation_qa',
#     'geometric_shapes',
#     'hyperbaton',
#     'logical_deduction_five_objects',
#     'logical_deduction_seven_objects',
#     'logical_deduction_three_objects',
#     'movie_recommendation',
#     'penguins_in_a_table',
#     'reasoning_about_colored_objects',
#     'ruin_names',
#     'salient_translation_error_detection',
#     'snarks',
#     'temporal_sequences',
#     'tracking_shuffled_objects_five_objects',
#     'tracking_shuffled_objects_seven_objects',
#     'tracking_shuffled_objects_three_objects',
# }

# def _bbh_extract_yesno(text: str) -> str:
#     match = re.search(r'\b(Yes|No)\b', text, re.IGNORECASE)
#     if match:
#         return match.group(1).capitalize()
#     return text.strip().split()[0] if text.strip() else ''

# def _bbh_extract_truefalse(text: str) -> str:
#     match = re.search(r'\b(True|False)\b', text, re.IGNORECASE)
#     if match:
#         return match.group(1).capitalize()
#     return text.strip().split()[0] if text.strip() else ''

# def _bbh_extract_valid_invalid(text: str) -> str:
#     match = re.search(r'\b(valid|invalid)\b', text, re.IGNORECASE)
#     if match:
#         return match.group(1).lower()
#     return text.strip().split()[0] if text.strip() else ''

# def _bbh_extract_numeric(text: str) -> str:
#     # Take the last number — models often reason first, then state the answer
#     matches = re.findall(r'-?\d+', text)
#     if matches:
#         return matches[-1]
#     return text.strip()

# def _bbh_extract_mcq(text: str) -> str:
#     patterns = [
#         r'(?i)answer\s*:\s*\(?([A-K])\)?',
#         r'(?i)the answer is\s*:?\s*\(?([A-K])\)?',
#         r'(?i)the correct answer is\s*:?\s*\(?([A-K])\)?',
#         r'\(([A-K])\)',
#         r'^([A-K])[\.:\s]',
#         r'([A-K])$',
#     ]
#     for pattern in patterns:
#         match = re.search(pattern, text, re.MULTILINE)
#         if match:
#             return f"({match.group(1).upper()})"
#     # Fallback: find any standalone capital letter A-K
#     match = re.search(r'\b([A-K])\b', text)
#     if match:
#         return f"({match.group(1).upper()})"
#     return text.strip()

# def _bbh_extract_freetext(text: str) -> str:
#     first_line = text.strip().split('\n')[0]
#     first_line = re.sub(r'^answer\s*:\s*', '', first_line,
#                         flags=re.IGNORECASE).strip()
#     return first_line

# @TEXT_POSTPROCESSORS.register_module('bbh')
# def bbh_postprocess(text: str, task_name: str = '') -> str:
#     """Comprehensive postprocessor for all 27 BBH tasks.

#     Dispatches to a specialised extractor based on the task's answer type:
#       - Yes/No       : causal_judgement, navigate, sports_understanding,
#                        web_of_lies
#       - True/False   : boolean_expressions
#       - valid/invalid: formal_fallacies
#       - Integer      : multistep_arithmetic_two, object_counting
#       - Free text    : dyck_languages, word_sorting
#       - MCQ (A-K)    : all remaining 17 tasks

#     When task_name is empty the answer type is inferred automatically from
#     the text content (useful for quick testing or unknown tasks).

#     Args:
#         text      : Raw model prediction string.
#         task_name : BBH task name, e.g. 'causal_judgement'.

#     Returns:
#         Cleaned, normalised answer string.

#     Usage in eval config::

#         bbh_eval_cfg = dict(
#             evaluator=dict(type=AccEvaluator),
#             pred_role='BOT',
#             pred_postprocessor=dict(type='bbh', task_name='causal_judgement'),
#         )
#     """
#     # Strip leading/trailing whitespace and drop everything after the first
#     # blank line (models frequently append explanations there).
#     text = text.strip()
#     first_block = re.split(r'\n\n', text)[0].strip()

#     # ── Task-name-based routing ──────────────────────────────────────────────
#     if task_name in _BBH_YESNO_TASKS:
#         return _bbh_extract_yesno(first_block)

#     if task_name in _BBH_TRUEFALSE_TASKS:
#         return _bbh_extract_truefalse(first_block)

#     if task_name in _BBH_VALID_INVALID_TASKS:
#         return _bbh_extract_valid_invalid(first_block)

#     if task_name in _BBH_NUMERIC_TASKS:
#         return _bbh_extract_numeric(first_block)

#     if task_name in _BBH_FREETEXT_TASKS:
#         return _bbh_extract_freetext(first_block)

#     if task_name in _BBH_MCQ_TASKS:
#         return _bbh_extract_mcq(first_block)

#     # ── Auto-detect fallback (task_name not provided / unrecognised) ─────────
#     if re.search(r'^\s*(Yes|No)\b', first_block, re.IGNORECASE):
#         return _bbh_extract_yesno(first_block)

#     if re.search(r'^\s*(True|False)\b', first_block, re.IGNORECASE):
#         return _bbh_extract_truefalse(first_block)

#     if re.search(r'^\s*(valid|invalid)\b', first_block, re.IGNORECASE):
#         return _bbh_extract_valid_invalid(first_block)

#     if re.match(r'^\s*-?\d+\s*$', first_block):
#         return _bbh_extract_numeric(first_block)

#     if re.match(r'^\s*\(?[A-K]\)?[\s\.\:]?$', first_block) or \
#        re.match(r'^\s*\(?[A-K]\)?[\s\.\:]', first_block):
#         return _bbh_extract_mcq(first_block)

#     return _bbh_extract_freetext(first_block)
# return match.group(1).upper()
