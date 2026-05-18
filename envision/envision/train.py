import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


def get_loaders(batch_size=128, data_root='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_root, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(data_root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
    return 100.0 * correct / len(loader.dataset)


def train(model, epochs=5, batch_size=128, lr=1e-3,
          save_path='outputs/model.pth', data_root='./data'):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = torch.device('cpu')
    model = model.to(device)

    train_loader, test_loader = get_loaders(batch_size, data_root)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining for {epochs} epochs on {len(train_loader.dataset):,} samples...")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Test Acc':>10}")
    print("-" * 32)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        acc = evaluate(model, test_loader, device)
        print(f"{epoch:>6}   {avg_loss:>11.4f}   {acc:>8.2f}%")

    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved -> {save_path}")
    return model, train_loader, test_loader
