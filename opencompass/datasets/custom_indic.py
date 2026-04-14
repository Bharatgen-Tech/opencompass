import json

import datasets

from opencompass.datasets.base import BaseDataset
from opencompass.registry import LOAD_DATASET


@LOAD_DATASET.register_module()
class CustomJsonlMCQDataset(BaseDataset):

    def load(self,
             path: str,
             question_key='question',
             options_keys=None,
             answer_key='answer',
             extra_keys=None,
             context_key=None,
             **kwargs):
        if options_keys is None:
            options_keys = ['A', 'B', 'C', 'D']

        extra_keys = extra_keys or []
        data_list = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)

                data_entry = {
                    'question': item.get(question_key, ''),
                    'answer': item.get(answer_key, ''),
                }
                for opt_key in options_keys:
                    data_entry[opt_key] = item.get(opt_key, '')

                for key in extra_keys:
                    data_entry[key] = item.get(key, '')

                if context_key:
                    data_entry['context'] = item.get(context_key, '')
                data_list.append(data_entry)

        # ✅ FIX: Convert the list of dicts to a Hugging Face Dataset
        # This provides the .map() method that OpenCompass is looking for
        dataset = datasets.Dataset.from_list(data_list)
        return dataset


@LOAD_DATASET.register_module()
class CustomJsonlPPLDataset(BaseDataset):

    def load(self, path: str, text_key='text'):
        data_list = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                data_list.append({'text': item.get(text_key, '')})

        # ✅ FIX: Convert to Hugging Face Dataset
        dataset = datasets.Dataset.from_list(data_list)
        return dataset


@LOAD_DATASET.register_module()
class CustomArrowMCQDataset(BaseDataset):

    def load(self,
             path: str,
             split='validation',
             question_key='question',
             choices_key='choices',
             answer_key='answer'):
        import datasets as hf_datasets

        ds = hf_datasets.load_from_disk(path)
        if isinstance(ds, hf_datasets.DatasetDict):
            ds = ds[split]

        options = ['A', 'B', 'C', 'D']

        def transform(item):
            choices = item[choices_key]
            return {
                'question': item[question_key],
                'A': choices[0],
                'B': choices[1],
                'C': choices[2],
                'D': choices[3],
                'answer': options[int(item[answer_key])],  # 1 -> 'B'
            }

        ds = ds.map(transform, remove_columns=ds.column_names)
        return ds
