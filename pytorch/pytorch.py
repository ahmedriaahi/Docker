import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load and transform the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Hyperparameters
epochs = 8
learning_rate = 0.003

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate model, define loss and optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training and testing functions
def train_model():
    model.train()
    train_loss = 0
    train_accuracy = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_accuracy += (predicted == labels).sum().item()

    train_loss /= len(train_loader)
    train_accuracy /= len(train_dataset)
    return train_loss, train_accuracy

def test_model():
    model.eval()
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_accuracy += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy /= len(test_dataset)
    return test_loss, test_accuracy

# Training loop
train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []

for epoch in range(epochs):
    print(f'\nEpoch {epoch+1}/{epochs}')
    train_loss, train_acc = train_model()
    test_loss, test_acc = test_model()
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

# Plot loss and accuracy over epochs
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Predictions on test set
model.eval()
with torch.no_grad():
    idx = np.random.randint(0, len(test_dataset), 16)
    test_images, test_labels = torch.stack([test_dataset[i][0] for i in idx]), np.array([test_dataset[i][1] for i in idx])

predictions = model(test_images)
_, predicted_labels = torch.max(predictions, 1)

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(test_images[i].squeeze(), cmap='gray')
    plt.title(f'Pred: {predicted_labels[i].item()}\nTrue: {test_labels[i]}')
    plt.axis('off')

plt.show()

# Save the model
torch.save(model.state_dict(), 'cnn_mnist.pth')

