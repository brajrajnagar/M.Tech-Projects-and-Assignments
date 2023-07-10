import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import sklearn.metrics as metrics
import joblib
import sys


script, test_data, outputfile = sys.argv

def load_data(filename):
    tagged_data = []
    with open(filename, 'r') as f:
        words = []
#         tags = []
        for line in f:
            if line == '\n':
                if words:
                    tagged_data.append(words)
                    words = []
#                     tags = []
            else:
                word = line.strip()
                words.append(word)
#                 tags.append(tag)
        if words:
            tagged_data.append(words)
    return tagged_data

test_tagged_lines = load_data(test_data)

# Set device to run on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define BiLSTM model
class BiLSTM_CNN(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, num_layers, output_size, dropout, 
                 char_vocab_size, char_embedding_size, kernel_size, padding_idx):
        super(BiLSTM_CNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, input_size, padding_idx=0)
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_size, padding_idx=padding_idx)
        self.conv1d = nn.Conv1d(char_embedding_size, input_size, kernel_size=kernel_size)
        self.lstm = nn.LSTM(input_size*2, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x, c):
        x = self.embedding(x)
        batch_size, seq_len, _ = c.shape
        c = c.reshape(batch_size * seq_len, -1)
        c = self.char_embedding(c)
        c = c.transpose(1, 2)
        c = self.conv1d(c)
        c, _ = torch.max(c, dim=2)
        c = c.reshape(batch_size, seq_len, -1)
        x = torch.cat([x, c], dim=-1)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
# Load the dictionary from the saved file
loaded_model_dict = joblib.load('aib222682')

# Extract the components from the dictionary
model = loaded_model_dict['model'].to(device)
word2idx = loaded_model_dict['word2idx']
ch2idx = loaded_model_dict['ch2idx']
label2idx = loaded_model_dict['label2idx']


class NERDataset(Dataset):
    def __init__(self, data, w2i, c2i, l2i, IsTraindataset):
        self.data = data
        self.IsTraindataset = IsTraindataset
        self.word2idx = w2i
        self.ch2idx = c2i
        self.label2idx = l2i
        self.max_len = max(len(d) for d in data)
        self.max_ch_len = max(len(ch) for d in data for ch in d)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]
        if self.IsTraindataset:
          sentence_idx = [self.word2idx[word] for word in sentence]
          character_idx = []
          for word in sentence:
            w = []
            for ch in word:
              w.append(self.ch2idx[ch])
            w += [self.ch2idx['< PAD >']] * (self.max_ch_len - len(w))
            character_idx.append(w)
        else:
          sentence_idx = [self.word2idx[word] if word in self.word2idx
                          else self.word2idx['< UNK >']
                          for word in sentence]
          character_idx = []
          for word in sentence:
            w = []
            for ch in word:
              if ch in self.ch2idx:
                w.append(self.ch2idx[ch])
              else:
                w.append(self.ch2idx['< UNK >'])
            # character padding
            w += [self.ch2idx['< PAD >']] * (self.max_ch_len - len(w))
            character_idx.append(w)
          # label_idx = [self.label2idx[label] for label in labels]

        # Pad sentence and labels with 0
        character_idx += [[0]*self.max_ch_len] * (self.max_len - len(sentence_idx))
        sentence_idx += [self.word2idx['< PAD >']] * (self.max_len - len(sentence_idx))
        # label_idx += [0] * (self.max_len - len(label_idx))

        return (torch.LongTensor(sentence_idx), torch.LongTensor(character_idx)) #torch.LongTensor(label_idx)
    

test_dataset = NERDataset(test_tagged_lines, word2idx, ch2idx, label2idx, IsTraindataset=False)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)




def evaluate(model, valid_loader, label2idx):
  final_outcome = []
  max_val_length = next(iter(valid_loader))[0].size()[1]
  model.eval()
  with torch.no_grad():
      for sentences in valid_loader:
          sentences_0, sentences_1 = sentences[0].to(device), sentences[1].to(device)
          output_logits = model(sentences_0, sentences_1)
          output = output_logits.argmax(dim=2)
          sentences_0, output = sentences_0.cpu().numpy(), output.cpu().numpy()
          for i, sentence in enumerate(sentences_0):
            words = []
            tags = []
            for j, word_logit in enumerate(sentence):
              if word_logit != 0:
                words.append(word_logit)
                tags.append(output[i,j])
                if j == max_val_length-1:
                  if label2idx['< PAD >'] in tags:
                    index = tags.index(label2idx['< PAD >'])
                    tags[index] = label2idx['S-Biological_Molecule']
                  final_outcome.append((words, tags))  
              else:
                if label2idx['< PAD >'] in tags:
                  index = tags.index(label2idx['< PAD >'])
                  tags[index] = label2idx['S-Biological_Molecule']
                final_outcome.append((words, tags))
                break

  idx2label = {value: key for key, value in label2idx.items()}

  with open(outputfile, 'w') as f:
    for i, tagged_sents in enumerate(final_outcome):
      _, labels = tagged_sents
      for label in labels:
        f.write("%s\n" %idx2label[label])
      if i < len(final_outcome)-1:
        f.write("\n")




EMBEDDING_SIZE = 200
HIDDEN_SIZE = 200
VOCAB_SIZE = len(word2idx)
NUM_LAYERS = 2
OUTPUT_SIZE = len(label2idx)
DROPOUT = 0.5
CHAR_VOCAB_SIZE = len(ch2idx)
CHAR_EMBEDDING_SIZE = 25
KERNEL_SIZE = 3
PADDING_IDX = 0
NUM_EPOCHS = 25
LEARNING_RATE = 0.001




evaluate(model, test_loader, label2idx)