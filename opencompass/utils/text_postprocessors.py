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

import re
def normalize_hindi_options(text: str) -> str:
    mapping = {
        'ए': 'A', 'A': 'A',
        'बी': 'B', 'B': 'B',
        'सी': 'C', 'C': 'C',
        'डी': 'D', 'D': 'D',
    }
    for k, v in mapping.items():
        text = text.replace(k, v)  # ✅ NO regex
    return text

def first_option_postprocess(text: str, options: str, cushion=True) -> str:
    """Find first valid option for text."""
    text = normalize_hindi_options(text)
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
