import numpy as np
import pandas as pd
import os
import sys
import torch
from torchvision.io import read_image
from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader


script, dataset_dir_path = sys.argv
# dataset_dir_path = '/kaggle/input/col774-2022'

train_X = pd.read_csv(os.path.join(dataset_dir_path, 'train_x.csv'))
train_y = pd.read_csv(os.path.join(dataset_dir_path, 'train_y.csv'))
test_X = pd.read_csv(os.path.join(dataset_dir_path, 'non_comp_test_x.csv'))
test_y = pd.read_csv(os.path.join(dataset_dir_path, 'train_y.csv')).iloc[:len(test_X),:]


class CustomImageDataset:
    def __init__(self, train_X, train_y, img_dir, transform=None, target_transform=None):
        self.img_name = train_X
        self.img_labels = train_y
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_name.iloc[idx, 1])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


tf=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
train_dataset = CustomImageDataset(train_X, train_y, os.path.join(dataset_dir_path, 'images/images'), transform=tf)
test_dataset = CustomImageDataset(test_X, test_y, os.path.join(dataset_dir_path, 'images/images'), transform=tf)


batch_size=128
train_loader = DataLoader(train_dataset, batch_size)
test_loader = DataLoader(test_dataset, batch_size)


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            #Input = 3 x 224 x 224, Output = 32 x 224 x 224
            torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5, padding = 2), 
            torch.nn.ReLU(),
            #Input = 32 x 224 x 224, Output = 32 x 112 x 112
            torch.nn.MaxPool2d(kernel_size=2),
  
            #Input = 32 x 112 x 112, Output = 64 x 112 x 112
            torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, padding = 2),
            torch.nn.ReLU(),
            #Input = 64 x 112 x 112, Output = 64 x 56 x 56
            torch.nn.MaxPool2d(kernel_size=2),
              
            #Input = 64 x 56 x 56, Output = 128 x 56 x 56
            torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5, padding = 2),
            torch.nn.ReLU(),
            #Input = 128 x 56 x 56, Output = 128 x 28 x 28
            torch.nn.MaxPool2d(kernel_size=2),
  
            torch.nn.Flatten(),
            torch.nn.Linear(128*28*28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 30)
        )
  
    def forward(self, x):
        return self.model(x)


#Selecting the appropriate training device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN().to(device)
  
#Defining the model hyper parameters
num_epochs = 30
learning_rate = 0.001
# weight_decay = 0.01
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Training process begins
train_loss_list = []
for epoch in range(num_epochs):
    # print(f'Epoch {epoch+1}/{num_epochs}:', end = ' ')
    train_loss = 0
      
    #Iterating over the training dataset in batches
    model.train()
    for i, (images, labels) in enumerate(train_loader):
          
        #Extracting images and target labels for the batch being iterated
        images = images.to(device)
        labels = labels.to(device)
        #Calculating the model output and the cross entropy loss
        outputs = model(images)
        loss = criterion(outputs, labels)
  
        #Updating weights according to calculated loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    #Printing loss for each epoch
    train_loss_list.append(train_loss/len(train_loader))
    # print(f"Training loss = {train_loss_list[-1]}")


test_acc=0
genre = np.array([])
model.eval()
  
with torch.no_grad():
    #Iterating over the training dataset in batches
    for i, (images, labels) in enumerate(test_loader):
          
        images = images.to(device)
        #y_true = labels.to(device)
          
        #Calculating outputs for the batch being iterated
        outputs = model(images)
          
        #Calculated prediction labels from models
        _, y_pred = torch.max(outputs.data, 1)
        genre = np.concatenate((genre, y_pred.cpu().detach().numpy()), axis=0)
          
        #Comparing predicted and true labels
        #test_acc += (y_pred == y_true).sum().item()
      
    # print(f"Test set accuracy = {100 * test_acc / len(test_dataset)} %")
    id = pd.read_csv(os.path.join(dataset_dir_path, 'non_comp_test_x.csv')).iloc[:,0]
    d = {'Id':np.array(id), 'Genre':genre}
    out = pd.DataFrame(data=d)
    out.to_csv('non_comp_test_pred_y.csv', index=False)
    # out.to_csv('/kaggle/working/non_comp_test_pred_y.csv', index=False)  
