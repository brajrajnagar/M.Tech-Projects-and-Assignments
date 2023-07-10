import torch
import os
import sys
import torchtext
import pandas as pd
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
from torch.utils.data import Dataset, DataLoader

data_path = sys.argv[1]
trainX = os.path.join(data_path, 'train_x.csv')
trainY = os.path.join(data_path, 'train_y.csv')
non_compX = os.path.join(data_path, 'non_comp_test_x.csv')
non_compY = os.path.join(data_path, 'non_comp_test_y.csv')
images_path = os.path.join(data_path, 'images/images')
compX = os.path.join(data_path, 'comp_test_x.csv')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = get_tokenizer('basic_english')
global_vectors = GloVe(name='6B', dim = 300)
seq_length = 15

class TextDataset(Dataset):
    def __init__(self, title_file, label_file, seq_length):
        self.title_file = pd.read_csv(title_file)
        self.label_file = pd.read_csv(label_file)
        self.max_length = seq_length
    
    def __len__(self):
        return len(self.title_file)
    
    def __getitem__(self,idx):
        title = self.title_file.iloc[idx, 2]
        title = title.lower()
        tokens = tokenizer(title)
        tokens = tokens+[""]*(self.max_length-len(tokens)) if len(tokens)<self.max_length else tokens[:self.max_length]
        title = global_vectors.get_vecs_by_tokens(tokens)
        label = self.label_file.iloc[self.title_file.iloc[idx,0],1]
        return title, label

class Model(torch.nn.Module):
    def __init__(self, seq_length):
        super(Model, self).__init__()
        self.rnn = torch.nn.RNN(input_size = 300, hidden_size = 128, batch_first = True, bidirectional = True)
        self.fc1 = torch.nn.Linear(seq_length*128*2,128)
        self.fc2 = torch.nn.Linear(128,30)
    
    def forward(self, data):
        title, label = data
        batch_size = title.shape[0]
        hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(title, hidden)
        out = out.reshape(batch_size,-1)
        x = self.fc1(out)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x
        
    def init_hidden(self, batch_size):
        hidden = torch.randn(2, batch_size, 128)
        return hidden.to(device)

def get_dataset():
    dataset = TextDataset(trainX, trainY, seq_length)
    train_set_length = int(len(dataset)*0.8)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_set_length, len(dataset)-train_set_length])
    trainloader = DataLoader(train_set, batch_size = 8192)
    valloader = DataLoader(val_set, batch_size = 1024)
    return trainloader, valloader

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    train_correct = 0
    train_loss = 0.0
    for (title, label) in dataloader:
        data = title.to(device), label.to(device)
        pred = model(data)
        loss = loss_fn(pred, data[1])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_correct += (pred.argmax(1)==data[1]).sum().item()
    print(f'Train loss: {(train_loss/len(dataloader)):>5f}', end = " ")
    print(f'Train accuracy: {(train_correct/size):>3f}')

def val_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    val_correct = 0
    val_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            title, label = data
            title, label = title.to(device), label.to(device)
            pred = model((title, label))
            loss = loss_fn(pred, label)
            
            val_loss += loss.item()
            val_correct += (pred.argmax(1)==label).sum().item()
    print(f'Val loss: {(val_loss/len(dataloader)):>5f}', end = " ")
    print(f'Val accuracy: {(val_correct/size):>3f}')

def train_model():
    trainloader, valloader = get_dataset()
    model = Model(seq_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay=5e-4)
    n_epochs = 15
    for epoch in range(n_epochs):
        print(f"Epoch: {epoch+1}/{n_epochs}")
        train_loop(trainloader, model, torch.nn.CrossEntropyLoss(), optimizer)
        val_loop(valloader, model, torch.nn.CrossEntropyLoss())
    return model

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_correct = 0
    test_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            title, label = data
            title, label = title.to(device), label.to(device)
            pred = model((title, label))
            loss = loss_fn(pred, label)
            
            test_loss += loss.item()
            test_correct += (pred.argmax(1)==label).sum().item()
    print(f'Test loss: {(test_loss/len(dataloader)):>5f}', end = " ")
    print(f'Test accuracy: {(test_correct/size):>3f}')

def generate_csv_file(dataloader, model):
    preds = []
    with torch.no_grad():
        for data in dataloader:
            title, label = data
            title, label = title.to(device), label.to(device)
            pred = model((title, label))
            preds.append(torch.argmax(pred, dim=1))
    out = torch.cat(preds)
    out = torch.unsqueeze(out, dim=1)
    out = out.to('cpu')
    ids = torch.arange(0,len(dataloader.dataset)).reshape(len(dataloader.dataset),-1)
    final = torch.cat([ids, out], dim = 1)
    df = pd.DataFrame(final.numpy(),columns=['Id','Genre'])
    df.to_csv("./non_comp_test_pred_y.csv",index=False)

def get_test_accuracy(model):
    test_set = TextDataset(non_compX, non_compY, seq_length)
    testloader = DataLoader(test_set, batch_size=1024)
    test_loop(testloader, model, torch.nn.CrossEntropyLoss())

def main():
    model = train_model()
    get_test_accuracy(model)
    test_set = TextDataset(non_compX, non_compY, seq_length)
    testloader = DataLoader(test_set, batch_size=1024)
    generate_csv_file(testloader, model)

if __name__ == '__main__':
    main()