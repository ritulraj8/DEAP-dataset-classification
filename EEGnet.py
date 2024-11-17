import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle

# Load DEAP dataset
def load_deap_data(data_dir):
    data = []
    labels = []

    for filename in os.listdir(data_dir):
        if filename.endswith('.dat'):
            with open(os.path.join(data_dir, filename), 'rb') as f:
                subject_data = pickle.load(f, encoding="latin1")
                eeg_data = subject_data['data']  # Shape: (40 trials, 40 channels, 8064 timepoints)
                eeg_labels = subject_data['labels']  # Shape: (40 trials, 4 labels)

                data.append(eeg_data)
                labels.append(eeg_labels)

    data = np.concatenate(data, axis=0)  # Combine all trials across subjects
    labels = np.concatenate(labels, axis=0)
    return data, labels

# Dataset class
class DEAPDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels[:, 1] > 5, dtype=torch.long)  # Binary classification for arousal

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eeg_data = self.data[idx]
        label = self.labels[idx]
        return eeg_data, label

# Display EEG data as a reshaped 2D grayscale image
def display_eeg_image(eeg_data, title="EEG Signal"):
    # Reshape timepoints into a grid based on channels and time segments
    num_channels, num_timepoints = eeg_data.shape
    reshaped_data = eeg_data[:, :400]  # Visualize the first 400 timepoints (20x20 grid per channel)
    reshaped_grid = reshaped_data.reshape(40, 20, 20)  # 40 channels, 20x20 time grid
    mean_image = reshaped_grid.mean(axis=0)  # Average across channels for visualization
    
    plt.imshow(mean_image, cmap='gray', aspect='auto')
    plt.title(title)
    plt.colorbar(label="Amplitude")
    plt.xlabel("Time segments (reshaped)")
    plt.ylabel("Channels (mean across)")
    plt.show()

# ResNet + LSTM model
class ResNetLSTM(nn.Module):
    def __init__(self, num_channels, num_timepoints, num_classes):
        super(ResNetLSTM, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()  # Remove the classifier head

        self.lstm = nn.LSTM(512, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.resnet(x)  # Extract features
        x = x.unsqueeze(1).repeat(1, x.size(1), 1)  # Prepare for LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take output from the last time step
        x = self.fc(x)
        return x

# Load and preprocess data
data_dir = 'C:/DEAP_classification/data_preprocessed_python/data_preprocessed_python'
data, labels = load_deap_data(data_dir)

# Normalize EEG data
data = (data - np.min(data)) / (np.max(data) - np.min(data))  # Min-max normalization

# Split into train and test sets
train_ratio = 0.8
num_samples = data.shape[0]
train_size = int(train_ratio * num_samples)
test_size = num_samples - train_size

train_data, test_data = data[:train_size], data[train_size:]
train_labels, test_labels = labels[:train_size], labels[train_size:]

# Create PyTorch datasets
train_dataset = DEAPDataset(train_data, train_labels)
test_dataset = DEAPDataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Display an input EEG sample
sample_eeg, _ = train_dataset[0]
display_eeg_image(sample_eeg.numpy(), title="Sample EEG Data")

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetLSTM(num_channels=40, num_timepoints=8064, num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

# Testing function
def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    return all_preds, all_labels

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=20)

# Test the model and visualize classification results
preds, labels = test_model(model, test_loader)

# Display a classified EEG sample
classified_sample = test_dataset[0][0].numpy()
predicted_label = preds[0]
display_eeg_image(classified_sample, title=f"Classified as: {predicted_label}")

