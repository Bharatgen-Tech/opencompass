import json
import datasets
from opencompass.registry import LOAD_DATASET
from opencompass.datasets.base import BaseDataset

@LOAD_DATASET.register_module()
class CustomJsonlMCQDataset(BaseDataset):
    def load(self, path: str, question_key='question', options_keys=None, answer_key='answer'):
        if options_keys is None:
            options_keys = ['A', 'B', 'C', 'D']
            
        data_list = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                item = json.loads(line)
                
                data_entry = {
                    'question': item.get(question_key, ''),
                    'answer': item.get(answer_key, ''),
                }
                for opt_key in options_keys:
                    data_entry[opt_key] = item.get(opt_key, '')
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
                if not line: continue
                item = json.loads(line)
                data_list.append({'text': item.get(text_key, '')})
        
        # ✅ FIX: Convert to Hugging Face Dataset
        dataset = datasets.Dataset.from_list(data_list)
        return dataset