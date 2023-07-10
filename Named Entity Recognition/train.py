import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import sklearn.metrics as metrics
import joblib
import copy
import sys

script, train_data, valid_data = sys.argv


def load_data(filename):
    tagged_data = []
    with open(filename, 'r') as f:
        words = []
        tags = []
        for line in f:
            if line == '\n':
                if words:
                    tagged_data.append((words, tags))
                    words = []
                    tags = []
            else:
                word, tag = line.strip().split('\t')
                words.append(word)
                tags.append(tag)
        if words:
            tagged_data.append((words, tags))
    return tagged_data

train_tagged_lines = load_data(train_data)
valid_tagged_lines = load_data(valid_data)


def create_vocab(data):
    # Create word and label vocabulary
    words = set(word for TaggedData in data for word in TaggedData[0])
    characters = set(ch for TaggedData in data for word in TaggedData[0] for ch in word)
    labels = set(label for TaggedData in data for label in TaggedData[1])

    # Add special tokens for padding and unknown words
    word2idx = {word: i+2 for i, word in enumerate(words)}
    word2idx['< PAD >'] = 0
    word2idx['< UNK >'] = 1
    ch2idx = {ch: i+2 for i, ch in enumerate(characters)}
    ch2idx['< PAD >'] = 0
    ch2idx['< UNK >'] = 1
    label2idx = {label: i+1 for i, label in enumerate(labels)}
    label2idx['< PAD >'] = 0 # use -1 as padding index for labels
    return word2idx, ch2idx, label2idx


word2idx, ch2idx, label2idx = create_vocab(train_tagged_lines)


class NERDataset(Dataset):
    def __init__(self, data, w2i, c2i, l2i, IsTraindataset):
        self.data = data
        self.IsTraindataset = IsTraindataset
        self.word2idx = w2i
        self.ch2idx = c2i
        self.label2idx = l2i
        self.max_len = max(len(d[0]) for d in data)
        self.max_ch_len = max(len(ch) for d in data for ch in d[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, labels = self.data[idx]
        if self.IsTraindataset:
          sentence_idx = [self.word2idx[word] for word in sentence]
          character_idx = []
          for word in sentence:
            w = []
            for ch in word:
              w.append(self.ch2idx[ch])
            w += [self.ch2idx['< PAD >']] * (self.max_ch_len - len(w))
            character_idx.append(w)
          label_idx = [self.label2idx[label] for label in labels]
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
          label_idx = [self.label2idx[label] for label in labels]

        # Pad sentence and labels with 0
        character_idx += [[0]*self.max_ch_len] * (self.max_len - len(sentence_idx))
        sentence_idx += [self.word2idx['< PAD >']] * (self.max_len - len(sentence_idx))
        label_idx += [0] * (self.max_len - len(label_idx))

        return (torch.LongTensor(sentence_idx), torch.LongTensor(character_idx)), torch.LongTensor(label_idx)
    

train_dataset = NERDataset(train_tagged_lines, word2idx, ch2idx, label2idx, IsTraindataset=True)
valid_dataset = NERDataset(valid_tagged_lines, word2idx, ch2idx, label2idx, IsTraindataset=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

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

# Train and evaluate the model
def train(model, optimizer, criterion, train_loader, valid_loader, num_epochs):
    best_valid_score = float('-inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for sentences, labels in train_loader:
            optimizer.zero_grad()
            sentences_0, sentences_1 = sentences[0].to(device), sentences[1].to(device)
            labels = labels.to(device)
            outputs = model(sentences_0, sentences_1)
            loss = criterion(outputs.view(len(labels.view(-1)), -1), labels.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        f1_micro, f1_macro = evaluate(model, valid_loader, label2idx)
        valid_score = (f1_micro + f1_macro)/2

        # print('Epoch: {:02d}, Train Loss: {:.4f}, Valid micro: {:.4f}, Valid macro: {:.4f}'.format(epoch+1, train_loss, f1_micro, f1_macro))

        if valid_score > best_valid_score:
            best_valid_score = valid_score
            # Create a copy of the model
            model_copy = copy.deepcopy(model)
            model_copy = BiLSTM_CNN(EMBEDDING_SIZE, HIDDEN_SIZE, VOCAB_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT,
                   CHAR_VOCAB_SIZE, CHAR_EMBEDDING_SIZE, KERNEL_SIZE, PADDING_IDX).to(device)
            model_copy.load_state_dict(model.state_dict())
            model_dict = {'model': model_copy.cpu(), 'word2idx':word2idx, 'ch2idx':ch2idx, 'label2idx':label2idx}
            joblib.dump(model_dict, 'aib222682')
            # print("model saved.......")


def evaluate(model, valid_loader, label2idx):
  final_outcome = []
  max_val_length = next(iter(valid_loader))[0][0].size()[1]
  model.eval()
  with torch.no_grad():
      for sentences, labels in valid_loader:
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

  pred = []
  for i, tagged_sents in enumerate(final_outcome):
    _, labels = tagged_sents
    for label in labels:
      pred.append(idx2label[label])

  possible_labels = ['O','B-Species', 'S-Species', 'S-Biological_Molecule', 'B-Chemical_Compound', 'B-Biological_Molecule', 'I-Species', 'I-Biological_Molecule', 'E-Species', 'E-Chemical_Compound', 'E-Biological_Molecule', 'I-Chemical_Compound', 'S-Chemical_Compound']

  def get_data_in(file_path):
      with open(file_path,"r")as fread:
          data = fread.readlines()

      tags = []
      for i,d in enumerate(data):
          d = d.replace("\n",'')
          if d == '':
              continue
          
          tag = d.split("\t")[-1]
          assert tag in possible_labels, f"Non-possible tag found : {tag} at line {i}"
          tags.append(tag)
      
      return tags


  pred_data = pred
  gold_data = get_data_in(valid_data)

  possible_labels.remove('O')
  f1_micro = metrics.f1_score(gold_data, pred_data, average="micro", labels=possible_labels)
  f1_macro = metrics.f1_score(gold_data, pred_data, average="macro", labels=possible_labels)
  return f1_micro, f1_macro


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


model = BiLSTM_CNN(EMBEDDING_SIZE, HIDDEN_SIZE, VOCAB_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT,
                   CHAR_VOCAB_SIZE, CHAR_EMBEDDING_SIZE, KERNEL_SIZE, PADDING_IDX).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=0)

train(model, optimizer, criterion, train_loader, valid_loader, NUM_EPOCHS)