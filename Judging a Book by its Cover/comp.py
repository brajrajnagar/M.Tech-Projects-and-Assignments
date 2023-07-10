import pandas as pd
import torch
import os,sys
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding

model_path = './best_model.pt'
data_path = sys.argv[1]
trainX = os.path.join(data_path, 'train_x.csv')
trainY = os.path.join(data_path, 'train_y.csv')
non_compX = os.path.join(data_path, 'non_comp_test_x.csv')
non_compY = os.path.join(data_path, 'non_comp_test_y.csv')
images_path = os.path.join(data_path, 'images/images')
compX = os.path.join(data_path, 'comp_test_x.csv')


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

class BookDataset(Dataset):
    def __init__(self, title_file, label_file, img_dir):
        self.title_file = pd.read_csv(title_file)
        self.img_dir = img_dir
        self.label_file = pd.read_csv(label_file)
        self.transforms = transforms.Compose([transforms.ConvertImageDtype(torch.float),
                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                             ])
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
    def __len__(self):
        return len(self.title_file)
    
    def __getitem__(self,idx):
        img_path = os.path.join(self.img_dir,self.title_file.iloc[idx,1])
        img = read_image(img_path)
        img = self.transforms(img)
        title = self.title_file.iloc[idx,2]
        label = self.label_file.iloc[self.title_file.iloc[idx,0],1]
        return img, title, label

class TestDataset(Dataset):
    def __init__(self, title_file, img_dir):
        self.title_file = pd.read_csv(title_file)
        self.img_dir = img_dir
        self.transforms = transforms.Compose([transforms.ConvertImageDtype(torch.float),
                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                             ])
        
    def __len__(self):
        return len(self.title_file)
    
    def __getitem__(self,idx):
        img_path = os.path.join(self.img_dir,self.title_file.iloc[idx,1])
        img = read_image(img_path)
        img = self.transforms(img)
        title = self.title_file.iloc[idx,2]
        return img, title

def collate(batch):
    images, titles, labels = list(zip(*batch))
    images = torch.cat(images, 0)
    labels = torch.tensor(labels)
    titles = tokenizer(list(titles))
    titles = data_collator(titles)
    return images, titles, labels

def get_dataset():
    dataset1 = BookDataset(trainX, trainY, images_path)
    dataset2 = BookDataset(non_compX, non_compY, images_path)
    dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
    train_set_length = int(len(dataset)*0.9)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_set_length, len(dataset)- train_set_length])
    trainloader = DataLoader(train_set, batch_size = 64, shuffle=True, collate_fn=collate)
    valloader = DataLoader(val_set, batch_size = 64, shuffle=True, collate_fn=collate)
    return trainloader, valloader

def val_check(dataloader, model):
    with torch.no_grad():
        correct = 0
        total_loss = 0.0
        for batch in dataloader:
            img, title, label = batch
            input_ids = title['input_ids'].to(device)
            attention_mask = title['attention_mask'].to(device)
            labels = label.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            correct += (outputs.logits.argmax(dim = 1)==labels).sum().item()
        print(f'Val loss: {total_loss/len(dataloader):>5f}', end = ' ')
        print(f'Val accuracy: {(correct/len(dataloader.dataset)):>3f}')
    return total_loss/len(dataloader)

def create_model():
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=30).to(device)
    return model

def train_model(model, trainloader, valloader):
    model.train()
    n_epochs = 5
    early_stopping = 5
    step_check = 50
    step_count = 0
    counter = 0
    val_best = float('inf')
    running_loss = 0.0
    optim = AdamW(model.parameters(), lr=5e-5)
    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1}:')
        print('-'*20)
        for batch, data in enumerate(trainloader):
            optim.zero_grad()
            img, title, label = data
            input_ids = title['input_ids'].to(device)
            attention_mask = title['attention_mask'].to(device)
            labels = label.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            running_loss += loss.item()
            loss.backward()
            optim.step()
            step_count+=1
            if step_count==step_check:
                print(f'Train_loss {(running_loss/step_count):>5f}')
                val_loss = val_check(valloader, model)
                if val_loss < val_best:
                    print(f'Val loss decreased {val_best} ----> {val_loss}')
                    val_best = val_loss
                    torch.save(model.state_dict(), model_path)
                    counter=0
                else:
                    counter+=1
                if counter == early_stopping:
                    break
                step_count = 0
                running_loss = 0.0
        if counter == early_stopping:
            break

def collate_test(batch):
    images, titles = list(zip(*batch))
    images = torch.cat(images, 0)
    titles = tokenizer(list(titles))
    titles = data_collator(titles)
    return images, titles

def generate_csv_file():
    best_model = create_model()
    best_model.load_state_dict(torch.load(model_path))
    test_set = TestDataset(compX, images_path)
    testloader = DataLoader(test_set, batch_size = 64, shuffle=False, collate_fn=collate_test)
    with torch.no_grad():
        preds = []
        best_model.eval()
        for batch, data in enumerate(testloader):
            img, title = data
            input_ids = title['input_ids'].to(device)
            attention_mask = title['attention_mask'].to(device)
            outputs = best_model(input_ids, attention_mask=attention_mask)
            preds.append(torch.argmax(outputs.logits,dim=1))
    out = torch.cat(preds)
    out = torch.unsqueeze(out, dim=1)
    out = out.to('cpu')
    ids = torch.arange(0,len(testloader.dataset)).reshape(len(testloader.dataset),-1)
    final = torch.cat([ids, out], dim = 1)
    df = pd.DataFrame(final.numpy(),columns=['Id','Genre'])
    df.to_csv("./comp_test_y.csv",index=False)

def main():
    trainloader, valloader = get_dataset()
    model = create_model()
    train_model(model, trainloader, valloader)
    generate_csv_file()

if __name__ == '__main__':
    main()