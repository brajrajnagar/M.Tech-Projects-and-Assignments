import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration
import re
import os
import sys

script, dev_PATH, pred_PATH = sys.argv


os.environ["TOKENIZERS_PARALLELISM"] = "true"

# print("testing....")

token_base = 't5-small'
model_base = 't5-small'


# Define the dataset class
class TODDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length):
        self.tokenizer = tokenizer
        self.data = data
        self.max_input_length = min(max_input_length+5, 1024)
        # self.max_target_length = min(max_target_length+5, 1024)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        input_text = example['input']
        input_text = re.sub(r'\b([uU]h+|[uU]m+|[uU]hm+|yeah|oh+|Oh+)\b', '', input_text)
        input_text = re.sub(r'\b(\w+(?:\s+\w+)*)\s+(\1)\b', r'\1', input_text)
        tokens = input_text.split()
        # Remove consecutive repeated words
        tokens = [token for i, token in enumerate(tokens) if i == 0 or token != tokens[i-1]]
        # Convert tokens back to a string
        input_text = ' '.join(tokens)
        input_text = re.sub(r'\b(\w+(?:\s+\w+)*)\s+(\1)\b', r'\1', input_text)
        
        history = ' '.join([h['user_query'] for h in example['history']])
        history2 = ' '.join([h['response_text'] for h in example['history']])
        user_contacts = ' '.join(example['user_contacts'])
        # full_input_text = input_text + ' ' + ' '.join([h['user_query'] for h in history]) + ' ' + ' '.join(user_contacts)
                
        input_tokens = self.tokenizer.batch_encode_plus(
            [input_text],
            padding = False,
            return_tensors='pt'
        )
        in_len = len(input_tokens['input_ids'].squeeze(dim=0))

        history_tokens = self.tokenizer.batch_encode_plus(
            [history],
            padding = False,
            return_tensors='pt'
        )
        his_len = len(history_tokens['input_ids'].squeeze(dim=0))
        
        history_tokens2 = self.tokenizer.batch_encode_plus(
            [history2],
            padding = False,
            return_tensors='pt'
        )
        his_len2 = len(history_tokens2['input_ids'].squeeze(dim=0))

        contact_tokens = self.tokenizer.batch_encode_plus(
            [user_contacts],
            max_length= self.max_input_length - in_len - his_len - his_len2,
            padding = 'max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        in_ids = torch.cat((input_tokens['input_ids'].squeeze(dim=0), 
                            history_tokens['input_ids'].squeeze(dim=0),
                            history_tokens2['input_ids'].squeeze(dim=0),
                            contact_tokens['input_ids'].squeeze(dim=0)
                            ))
        att_mask = torch.cat((input_tokens['attention_mask'].squeeze(dim=0), 
                            history_tokens['attention_mask'].squeeze(dim=0),
                            history_tokens2['attention_mask'].squeeze(dim=0),
                            contact_tokens['attention_mask'].squeeze(dim=0)
                            ))

        # output_text = example.get('output', None)
        # output_tokens = self.tokenizer.batch_encode_plus(
        #     [output_text],
        #     max_length= self.max_target_length,
        #     padding = 'max_length',
        #     truncation=True,
        #     return_tensors='pt'
        # ) if output_text is not None else None
        return {
            'input_ids': in_ids,
            'attention_mask': att_mask
        }
    

# Define the data collator function
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(token_base)
model = T5ForConditionalGeneration.from_pretrained(model_base)
model.load_state_dict(torch.load('aib222682_aib222671_model.pth'))


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
model.to(device)


random.seed(10)
max_in_len, max_out_len = 0, 0

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

test_data = load_data(dev_PATH)
max_test_in = 0
for item in test_data:
    l_input = item['input']
    l_history = ' '.join([h['user_query'] for h in item['history']])
    l_history2 = ' '.join([h['response_text'] for h in item['history']])
    l_contacts = ' '.join(item['user_contacts'])
    # l_out = item['output']
    # l_total = l_input +' ' + l_history+' ' + l_contacts
    t1 = tokenizer.batch_encode_plus(
            [l_input],
            padding = False,
            return_tensors='pt'
        )
    if t1['input_ids'].dim()==0:
        t1['input_ids'] = torch.LongTensor([])
    t2 = tokenizer.batch_encode_plus(
            [l_history],
            padding = False,
            return_tensors='pt'
        )
    if t2['input_ids'].dim()==0:
        t2['input_ids'] = torch.LongTensor([])
    t3 = tokenizer.batch_encode_plus(
            [l_history2],
            padding = False,
            return_tensors='pt'
        )
    if t3['input_ids'].dim()==0:
        t3['input_ids'] = torch.LongTensor([])
    t4 = tokenizer.batch_encode_plus(
            [l_contacts],
            padding = False,
            return_tensors='pt'
        )
    if t4['input_ids'].dim()==0:
        t4['input_ids'] = torch.LongTensor([])
    # out_tokn = tokenizer.batch_encode_plus(
    #         [l_out],
    #         padding = False,
    #         return_tensors='pt'
    #     )
    # if out_tokn['input_ids'].dim()==0:
    #     out_tokn['input_ids'] = torch.LongTensor([])

    L_in = len(t1['input_ids'].squeeze(dim=0)) + len(t2['input_ids'].squeeze(dim=0)) + len(t3['input_ids'].squeeze(dim=0)) + len(t4['input_ids'].squeeze(dim=0))
    # L_out = len(out_tokn['input_ids'].squeeze(dim=0))

    if L_in > max_test_in:
        max_test_in = L_in
    # if L_out > max_test_out:
    #     max_test_out = L_out


test_dataset = TODDataset(test_data, tokenizer, max_test_in)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= 60, shuffle=False, 
                                          collate_fn=collate_fn)

# print("Inferencing.....", flush=True)
model.eval()
pred_strs = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model.generate(input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=150,
                    num_beams=3,
                    early_stopping=True)
        # Convert output tokens to string format
        output_strs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        pred_strs += output_strs
        torch.cuda.empty_cache()
        
del input_ids, attention_mask, outputs, output_strs

# Evaluation              
# data = load_jsonl(data_file)
# preds = pred_strs

# metrics = compute_metrics(data, preds)
# print(json.dumps(metrics, indent=2), flush=True)

with open(pred_PATH, 'w') as file:
    # Write each string to the file
    for string in pred_strs:
        file.write(string + '\n')
