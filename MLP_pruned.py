import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import torch.nn.utils.prune as prune
import csv
import time
import psutil
import gc
import torch.profiler
import torch.nn.functional as F
from torch.utils.benchmark import Timer

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(85, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(128, 83)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        y = self.decoder(x)
        return y

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)

def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory Usage: {memory_info.rss / (1024 * 1024):.2f} MB")

def train(model, train_loader, optimizer, scheduler):
    model.train()

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = torch.sqrt(nn.MSELoss()(output, target) + 1e-6)
        loss.backward()
        optimizer.step()
        scheduler.step()

def evaluate(model, val_loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += torch.sqrt(nn.MSELoss()(output, target) + 1e-6).item()
    return val_loss / len(val_loader)

def evaluate_bias(model, X, y):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32)
        predictions = model(inputs).numpy()
    differences = predictions - y
    if len(differences) == 0:
        return 0
    else:
        mean_bias = np.mean(differences)
        return mean_bias

# Load dataset
dataset = np.loadtxt("/Users/kas/Desktop/EDFA_ML-main/EDFA_1_18dB.csv", delimiter=",")
X = dataset[:, 1:86]
y = dataset[:, 86:]

# Initialize model, optimizer, scheduler
model = NN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08)
scheduler = CosineAnnealingLR(optimizer, 20, 1e-5)
kf = KFold(n_splits=5, shuffle=True, random_state=0)
prune_percentage = 50

# Track best loss
best_loss = float('inf')

# Append validation loss to the list
ls = []

# Training loop
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    train_dataset = MyDataset(X[train_index], y[train_index])
    val_dataset = MyDataset(X[val_index], y[val_index])
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    # トータルパラメータ数
    total_parameters_before_pruning = sum(p.numel() for p in model.parameters())
    print("Before pruning - Total parameters:", total_parameters_before_pruning)
    for epoch in range(500):
        train(model, train_loader, optimizer, scheduler)
        val_loss = evaluate(model, val_loader)

        # Print validation loss
        print(f"Fold {fold}, Epoch {epoch + 1}, Val Loss: {val_loss:.4f}")
        for param in model.parameters():
            del param
        # すべての世代のガベージコレクションを実行
        gc.collect()
        # Print memory usage
        get_memory_usage()
        # Timer setup
        timer_before_pruning = Timer(
            stmt="model(data)",
            setup="from __main__ import model, train_loader; data, _ = next(iter(train_loader))"
        )
        time_before_pruning = timer_before_pruning.timeit(100)
        print(f"Inference time before pruning: {time_before_pruning} seconds")
        # Prune at specified epochs
        if epoch in [350, 400, 450]:
            # Measure inference time before pruning
            timer_before_pruning = Timer(
                stmt="model(input_data)",
                setup="from __main__ import model, input_data",
            )

            # プルーニング前の各層のスパース性を確認
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    weights = module.weight.data
                    sparsity = float(torch.sum(weights == 0) / weights.numel())
                    print(f"Layer {name} Sparsity before pruning: {sparsity:.4f}")
                    print(f"Layer {name} Weights before pruning:\n{weights}")

            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    weights = module.weight.data
                    amount = prune_percentage / 100.0
                    prune.l1_unstructured(module, name='weight', amount=amount)
                    # プルーニング後のモデル
                    model_after_pruning = NN()


                    # プルーニング後の各層のスパース性を確認
                    for name, module in model_after_pruning.named_modules():
                        if isinstance(module, nn.Linear):
                            weights = module.weight.data
                            print(f"Layer {name} Weights after pruning:\n{weights}")
                            sparsity = float(torch.sum(weights == 0) / weights.numel())
                            print(f"Layer {name} Sparsity after pruning: {sparsity:.4f}")

                    # Print total parameters after pruning
                    print("Total parameters:", sum(p.numel() for p in model.parameters()))

                    # プルーニング前のモデルとプルーニング後のモデルのサイズ比較
                    print(f"Model size before pruning: {sum(p.numel() for p in model.parameters())} parameters")
                    print(
                        f"Model size after pruning: {sum(p.numel() for p in model_after_pruning.parameters())} parameters")

                    timer_after_pruning = Timer(
                        stmt="model(data)",
                        setup="from __main__ import model, train_loader; data, _ = next(iter(train_loader))"
                    )
                    time_after_pruning = timer_after_pruning.timeit(100)
                    print(f"Inference time after pruning: {time_after_pruning} seconds")
        # Save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            filename = f'model_best_{fold}_{epoch + 1}.pth'
            torch.save(model.state_dict(), filename)
            print(f"********************************BEST {fold}-RMSE: {best_loss:.2e}*****************************")
        if epoch == 499:
            ls.append(best_loss)

# Save the loss values to a CSV file
p = pd.DataFrame(ls)
filename = 'loss_pruned.csv'
p.to_csv(filename)

get_memory_usage()

