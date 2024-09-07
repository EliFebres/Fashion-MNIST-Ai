import torch

from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from Utility.cnn_model import FashionMNISTModel
from Utility.utility import train_and_test_model


device = "cuda" if torch.cuda.is_available() else "cpu"

# Download Data from torchvision.datasets
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
class_names = train_data.classes

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
epochs = 25
train_and_test_model(model, train_dataloader, test_dataloader, loss_fn, optimizer, device, epochs)

# Export Model
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "fashion_mnist_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)