import torch
from torch.utils.data import DataLoader

def calc_accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

def train_and_test_model(model: torch.nn.Module,
                         train_dataloader: DataLoader,
                         test_dataloader: DataLoader,
                         loss_fn: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         device: torch.device,
                         epochs):

    model.to(device)
    
    for epoch in range(epochs):
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