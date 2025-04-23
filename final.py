import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# === Step 1: Prepare data transforms ===
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),  # ✅ Change to 256x256
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (1.0,))
])

# === Step 2: Load dataset ===
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, 'dataset')
dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

print("Classes found:", dataset.classes)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# === Step 3: Define the neural network ===
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        input_size = 256 * 256  # ✅ Adjusted to 256x256

        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 4)

    def forward(self, x):
        x = x.view(-1, 256 * 256)  # ✅ Adjust input size
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = SimpleNN()

# === Step 4: Load existing model if available ===
if os.path.exists('trained_model.pth'):
    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()
    print("Loaded existing trained model!")
else:
    print("No saved model found, training from scratch.")

# === Step 5: Define loss and optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === Step 6: Train the model ===
epochs = 30
train_losses = []
train_accuracies = []

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.2f}%")

print("Training completed!")

# === Step 7: Save the model ===
torch.save(model.state_dict(), 'trained_model.pth')
print("Model saved successfully!")

# === Step 8: Test the model ===
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Sample predictions
print("\nSample predictions on test set:")
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        print(f"True: {dataset.classes[labels.item()]} | Predicted: {dataset.classes[predicted.item()]}")
        if i >= 10:
            break

# === Step 9: Plot training curves and save to file ===
epochs_range = range(1, epochs + 1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
print("Training curves saved as 'training_curves.png'")
