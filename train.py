# train.py
import torch
import torch.optim as optim
import torch.nn as nn
from model import SignLanguageCNN  # Import the CNN model
from dataLoader import get_data_loaders  # Import the custom data loader

# Get the data loaders
train_loader, test_loader = get_data_loaders()

# Initialize model, loss function, and optimizer
model = SignLanguageCNN()  # Your CNN model
criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()  # Clear the gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update the weights
        running_loss += loss.item()  # Accumulate the loss
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

# Evaluate the model
correct = 0
total = 0
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)  # Forward pass for evaluation
        _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()  # Compare predictions with actual labels

print(f'Accuracy: {100 * correct / total}%')

# Save the trained model
torch.save(model.state_dict(), 'saved_models/sign_language_model.pth')
