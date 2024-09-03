import random
import torch

from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from Utility.utility import calc_accuracy
from Utility.visualizer import single_image
from Utility.cnn_model import FashionMNISTModel


device = "cuda" if torch.cuda.is_available() else "cpu"

# Download Data from torchvision.datasets
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
class_names = train_data.classes

# Visualize a random image
# random_idx = random.randint(0, len(train_data) - 1)
# image, label = train_data[random_idx]
# label = class_names[label]
# single_image(image, label) # Visualize a random image from the training data

# Use dataloader to creat batches for forward passes
BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

# Create Model, Optimizer, and Loss Function
model = FashionMNISTModel(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

# Train Model
torch.manual_seed(42)
EPOCHS = 10

for epoch in range(EPOCHS):
    print(f'Epoch: {epoch + 1}')
    
    train_loss, train_acc = 0, 0
    model.train()  # Set the model to training mode
    
    for batch, (X, y) in enumerate(train_dataloader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)
        
        # 1. Forward Pass
        y_pred = model(X)
        
        # 2. Calculate Loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        
        # 3. Optimizer zero grad
        optimizer.zero_grad()
        
        # 4. Loss backward
        loss.backward()
        
        # 5. Optimizer step
        optimizer.step()
        
        # Calculate accuracy
        train_acc += calc_accuracy(y_true=y, y_pred=y_pred.argmax(dim=1))
    
    # Testing
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(test_dataloader):
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            
            # 3. Calculate Loss
            test_acc += calc_accuracy(y_true=y, y_pred=test_pred.argmax(dim=1))
            
    # Divide by number of batches to get average loss and accuracy
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)
            
    # Print out what's happening
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}% | Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%")
    print('-' * 20)