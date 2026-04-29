from datasets import load_dataset

ds = load_dataset('ai4bharat/hellaswag-translated', 'hi')
print(ds['validation'][0])