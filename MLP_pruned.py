#成功したもの
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import torch.nn.utils.prune as prune
import csv
import time
import matplotlib.pyplot as plt
import torch.autograd as autograd
import psutil
import gc
from torch.utils.benchmark import Timer
import torch.nn.functional as F

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
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)

start_time = time.time()
dataset = np.loadtxt("/Users/kas/Desktop/EDFA_ML-main/EDFA_1_18dB.csv", delimiter=",")
X = dataset[:, 1:86]
y = dataset[:, 86:]

def get_memory_usage():
    # プロセスのメモリ使用量を取得
    process = psutil.Process()
    memory_info = process.memory_info()
    # メモリ使用量を表示
    print(f"Memory Usage: {memory_info.rss / (1024 * 1024):.2f} MB")
def train(model, train_loader, optimizer):
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

model = NN()

def evaluate_bias(model, X, y):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32)
        predictions = model(inputs).numpy()  # テンソルをNumPy配列に変換
    errors = predictions - y
    if len(errors) == 0:
        return 0
    else:
        mean_bias = np.mean(errors)
        return mean_bias

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08)
scheduler = CosineAnnealingLR(optimizer, 20, 1e-5)
loss_fn = nn.MSELoss()
kf = KFold(n_splits=5, shuffle=True, random_state=0)
ls = []
params = model.state_dict()
print("Weights of layer1:", params['layer1.0.weight'])
prune_percentage = 50
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    train_dataset = MyDataset(X[train_index], y[train_index])
    val_dataset = MyDataset(X[val_index], y[val_index])
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    # トータルパラメータ数
    total_parameters_before_pruning = sum(p.numel() for p in model.parameters())
    print("Before pruning - Total parameters:", total_parameters_before_pruning)

    for epoch in range(500):
        train(model, train_loader, optimizer)
        val_loss = evaluate(model, val_loader)
        print("Fold {}, Epoch {}, Val Loss: {:.4f}".format(fold, epoch + 1, val_loss))
        # すべての世代のガベージコレクションを実行
        gc.collect()
        get_memory_usage()

        if fold == 0 and epoch == 0:
            best_loss = val_loss
        elif epoch % 50 == 0:  # 50エポックごとにプルーニング
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    # プルーニング前の各層のスパース性を確認
                    weights = module.weight.data
                    sparsity = float(torch.sum(weights == 0) / weights.numel())
                    print(f"Layer {name} Sparsity before pruning: {sparsity:.4f}")
                    print(f"Layer {name} Weights before pruning:\n{weights}")

                    # プルーニング
                    row_norms = torch.norm(weights, p=1, dim=1)
                    threshold = torch.kthvalue(row_norms, int(row_norms.size(0) * (prune_percentage / 100.0)))[0]
                    masks = (row_norms >= threshold).float().view(weights.size(0), 1)
                    module.weight.data = weights * masks

                    # プルーニング後の各層のスパース性を確認
                    sparsity = float(torch.sum(module.weight.data == 0) / module.weight.data.numel())
                    print(f"Layer {name} Sparsity after pruning: {sparsity:.4f}")
                    print(f"Layer {name} Weights after pruning:\n{module.weight.data}")

            # トータルパラメータ数
            total_parameters_after_pruning = sum(p.numel() for p in model.parameters())
            print("Total parameters after pruning:", total_parameters_after_pruning)

        elif val_loss < best_loss:
            best_loss = val_loss
            filename = 'model_best_{}-{}.pth'.format(fold, epoch + 1)
            torch.save(model.state_dict(), filename)
            print("********************************BEST {}-RMSE: {:.2e}*****************************".format(fold,
                                                                                                             best_loss))

        if epoch == 499:
            ls.append(best_loss)

# Save the loss values to a CSV file
p = pd.DataFrame(ls)
filename = 'loss_pruned.csv'
p.to_csv(filename)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"処理にかかった時間: {elapsed_time}秒")
get_memory_usage()

