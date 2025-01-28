import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import  DataLoader, random_split
import torcheeg
from torcheeg.datasets import DEAPDataset
from torcheeg import transforms
from torcheeg.models import EEGNet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
import seaborn as sns

from torcheeg.datasets.constants import \
    DEAP_CHANNEL_LOCATION_DICT

dataset = DEAPDataset(
    io_path=r'C:\DEAP_classification\.torcheeg\datasets_1738005784587_URCdH',
    root_path='C:/DEAP_classification/data_preprocessed_python/data_preprocessed_python',
                      online_transform=transforms.Compose([
                          transforms.To2d(),
                          transforms.ToTensor(),
                          transforms.RandomNoise(mean=0.0, std=0.1),
                      ]),
                      label_transform=transforms.Compose([
                          transforms.Select('valence'),
                          transforms.Binary(5.0),
                      ]))


train_size=int(0.8*len(dataset))
test_size=len(dataset)-train_size

train_dataset,test_dataset=random_split(dataset,[train_size,test_size])

#Hyperparameters
batch_size=32
learning_rate=0.0005
num_epochs=100

train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

images,labels=next(iter(train_loader))
imshow(torchvision.utils.make_grid(images))

model=EEGNet(chunk_size=128,num_electrodes=32,F1=16,F2=32,D=2,num_classes=2)

classes=('Low Valence','High Valence')

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

n_total_steps=len(train_loader)

for epoch in range(num_epochs):
    running_loss=0.0
    model.train()

    for i,(inputs,labels) in enumerate(train_loader):
        inputs=inputs.to(device)
        labels=labels.to(device)

        outputs=model(inputs)
        loss=criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if(i+1)%1000==0:
            print(f'Epoch [{epoch+1}/{num_epochs}],Loss: {loss.item()}')

print("Finished Training")

model.eval()
all_preds=[]
all_labels=[]

with torch.no_grad():
    for images,labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        _,predicted=torch.max(outputs,1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds=np.array(all_preds)
all_labels=np.array(all_labels)

precision=precision_score(all_labels,all_preds,average=None)
recall=recall_score(all_labels,all_preds,average=None)
f1=f1_score(all_labels,all_preds,average=None)
accuracy=accuracy_score(all_labels,all_preds)

print(f'Accuracy: {accuracy * 100:.2f}%')

confusion_matrix=confusion_matrix(all_labels,all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')





























