import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration
import re
import os
import sys

script, train_PATH, dev_PATH = sys.argv

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# print("training....")

token_base = 't5-small'
model_base = 't5-small'
# train_PATH = 'data/train.jsonl'
# dev_PATH = 'data/dev.jsonl'
# pred_PATH = 'pred.txt'


# Define the dataset class
class TODDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length, max_target_length):
        self.tokenizer = tokenizer
        self.data = data
        self.max_input_length = min(max_input_length+5, 1024)
        self.max_target_length = min(max_target_length+5, 1024)

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

        output_text = example.get('output', None)
        output_tokens = self.tokenizer.batch_encode_plus(
            [output_text],
            max_length= self.max_target_length,
            padding = 'max_length',
            truncation=True,
            return_tensors='pt'
        ) if output_text is not None else None
        return {
            'input_ids': in_ids,
            'attention_mask': att_mask,
            'output_ids': output_tokens['input_ids'].squeeze() if output_tokens else None,
            'output_attention_mask': output_tokens['attention_mask'].squeeze() if output_tokens else None,
        }
    


# Define the data collator function
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    output_ids = torch.stack([item['output_ids'] for item in batch if item['output_ids'] is not None])
    output_attention_mask = torch.stack([item['output_attention_mask'] for item in batch if item['output_attention_mask'] is not None])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'output_ids': output_ids,
        'output_attention_mask': output_attention_mask
    }



# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(token_base)
model = T5ForConditionalGeneration.from_pretrained(model_base)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
model.to(device)


random.seed(10)
max_in_len, max_out_len = 0, 0

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

train_data = load_data(train_PATH)
# train_outputs = []
max_train_in, max_train_out = 0, 0
for item in train_data:
    l_input = item['input']
    l_history = ' '.join([h['user_query'] for h in item['history']])
    l_history2 = ' '.join([h['response_text'] for h in item['history']])
    l_contacts = ' '.join(item['user_contacts'])
    l_out = item['output']
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
    out_tokn = tokenizer.batch_encode_plus(
            [l_out],
            padding = False,
            return_tensors='pt'
        )
    if out_tokn['input_ids'].dim()==0:
        out_tokn['input_ids'] = torch.LongTensor([])

    L_in = len(t1['input_ids'].squeeze(dim=0)) + len(t2['input_ids'].squeeze(dim=0)) + len(t3['input_ids'].squeeze(dim=0)) + len(t4['input_ids'].squeeze(dim=0))
    L_out = len(out_tokn['input_ids'].squeeze(dim=0))

    if L_in > max_train_in:
        max_train_in = L_in
    if L_out > max_train_out:
        max_train_out = L_out
        
test_data = load_data(dev_PATH)
test_outputs = []
max_test_in, max_test_out = 0, 0
for item in test_data:
    test_outputs.append(item['output'])

    l_input = item['input']
    l_history = ' '.join([h['user_query'] for h in item['history']])
    l_history2 = ' '.join([h['response_text'] for h in item['history']])
    l_contacts = ' '.join(item['user_contacts'])
    l_out = item['output']
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
    out_tokn = tokenizer.batch_encode_plus(
            [l_out],
            padding = False,
            return_tensors='pt'
        )
    if out_tokn['input_ids'].dim()==0:
        out_tokn['input_ids'] = torch.LongTensor([])

    L_in = len(t1['input_ids'].squeeze(dim=0)) + len(t2['input_ids'].squeeze(dim=0)) + len(t3['input_ids'].squeeze(dim=0)) + len(t4['input_ids'].squeeze(dim=0))
    L_out = len(out_tokn['input_ids'].squeeze(dim=0))

    if L_in > max_test_in:
        max_test_in = L_in
    if L_out > max_test_out:
        max_test_out = L_out

del l_input,l_history, l_history2, l_contacts, l_out, t1, t2, t3, t4, out_tokn, L_in, L_out
random.shuffle(train_data)

# print(f"{len(train_data)} {len(test_data)}", flush=True)
# print(f"{max_train_in} {max_train_out} {max_test_in} {max_test_out}", flush=True)


# Load the training and testing datasets
train_dataset = TODDataset(train_data, tokenizer, max_train_in, max_train_out)
test_dataset = TODDataset(test_data, tokenizer, max_test_in, max_test_out)


def parse(tokens):
    if "(" not in tokens:
        assert ")" not in tokens
        ret = dict()
        start = 0
        mid = 0
        for ii, tok in enumerate(tokens):
            if tok == "«":
                mid = ii
            elif tok == "»":
                key = ' '.join(tokens[start:mid])
                val = ' '.join(tokens[mid + 1:ii])
                ret[key] = val
                start = mid = ii + 1
        return ret

    st = tokens.index("(")
    outer_key = ' '.join(tokens[0:st])
    assert tokens[-1] == ")", " ".join(tokens)

    level = 0
    last = st + 1
    ret = dict()
    for ii in range(st + 1, len(tokens) - 1, 1):
        tok = tokens[ii]
        if tok == "»" and level == 0:
            rr = parse(tokens[last:ii + 1])
            ret.update(rr)
            last = ii + 1
        elif tok == "(":
            level += 1
        elif tok == ")":
            level -= 1
            if level == 0:
                rr = parse(tokens[last:ii + 1])
                ret.update(rr)
                last = ii + 1

    return {outer_key: ret}


def load_jsonl(fname):
    data = []
    with open(fname, 'r', encoding='utf-8') as fp:
        for line in fp:
            data.append(json.loads(line.strip()))

    return data


def per_sample_metric(gold, pred):
    ret = dict()
    ret['accuracy'] = int(gold == pred)

    get_intent = lambda x: x.split('(', 1)[0].strip()
    gintent = get_intent(gold)
    pintent = get_intent(pred)
    ret['intent_accuracy'] = int(gintent == pintent)

    parse_correct = 1
    try:
        _ = parse(pred.split())
    except:
        parse_correct = 0
    ret['parsing_accuracy'] = parse_correct

    return ret


def compute_metrics(data, preds):
    assert len(data) == len(preds), "Different number of samples in data and prediction."

    golds = [x['output'] for x in data]

    metrics = [per_sample_metric(gold, pred) for gold, pred in zip(golds, preds)]
    final_metrics = dict()
    mnames = list(metrics[0].keys())
    for key in mnames:
        final_metrics[key] = sum([met[key] for met in metrics]) / len(golds)
    
    return final_metrics


data_file = dev_PATH
# pred_file = pred_PATH



# Define the training parameters
batch_size = 30
num_epochs = 30
learning_rate = np.linspace(5e-4,1e-5,num_epochs)


# Define the optimizer and the loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate[0])
criterion = torch.nn.CrossEntropyLoss()


# Train the model
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, 
                                           shuffle=True, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= 60, shuffle=False, 
                                          collate_fn=collate_fn)



prev_metric_score = 0



for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for i, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output_ids = batch['output_ids'].to(device)
        output_attention_mask = batch['output_attention_mask'].to(device)
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=output_ids,
            decoder_attention_mask=output_attention_mask,
            use_cache=False
        )
        loss = criterion(outputs.logits.view(-1, outputs.logits.shape[-1]), output_ids.view(-1))
        train_loss += loss.item()
        # Backward and optimize
        loss.backward()
        optimizer.step()
        #Print loss every 300 batches
        # if (i + 1) % 300 == 0:
            # print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], AvgLoss: {train_loss/(i+1)}', flush=True)
        torch.cuda.empty_cache()
    
    del input_ids, attention_mask, output_ids, output_attention_mask, outputs, loss, train_loss

    if epoch > 14 and (epoch+1)%2==0:
        # print("Inferencing.....", flush=True)
        model.eval()
        pred_strs = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_length=max_train_out,
                            num_beams=3,
                            early_stopping=True)
                # Convert output tokens to string format
                output_strs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                pred_strs += output_strs
                torch.cuda.empty_cache()
                
        del input_ids, attention_mask, outputs, output_strs
        
        # Evaluation              
        data = load_jsonl(data_file)
        preds = pred_strs

        metrics = compute_metrics(data, preds)
        # print(json.dumps(metrics, indent=2), flush=True)
        
        scores = metrics.values()
        avrg = sum(scores)/len(scores)
        if avrg > prev_metric_score:
            torch.save(model.state_dict(), 'aib222682_aib222671_model.pth')
            # print('model saved .....', flush=True)
        #     with open(pred_PATH, 'w') as file:
        #         # Write each string to the file
        #         for string in pred_strs:
        #             file.write(string + '\n')
        #     prev_metric_score = avrg
        # code_end = time.time()
        # print(f"Time taken till now {(code_end - code_start)/60} min.\n", flush=True)
        torch.cuda.empty_cache()

    if epoch < num_epochs-1:
        # Update the learning rate for the optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate[epoch+1]



