import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import numpy as np
from model import NN
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import csv
#import matplotlib.pyplot as plt

# Define the dataset class

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)


dataset = np.loadtxt(r"C:\Users\Administrator\Desktop\EDFA-NN\EDFA_1_18dB.csv", delimiter=",")
X = dataset[:, 1:86]
y = dataset[:, 86:]


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
print(model)

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

# バイアスの評価
bias = evaluate_bias(model, X, y)

print("Mean Bias:", bias)

# Calculate mean and variance of input features
mean = np.mean(X, axis=0)
variance = np.var(X, axis=0)

print("Mean of input features:", mean)
print("Variance of input features:", variance)


# Define output file path
output_file = r"C:\Users\Administrator\Desktop\EDFA-NN\statistics.csv"

# Prepare data to write to CSV file
data = [['Feature', 'Mean', 'Variance']]
for i, (m, v) in enumerate(zip(mean, variance)):
    data.append([f'Feature {i+1}', m, v])

# Write data to CSV file
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print("CSV file saved successfully.")


optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08)
scheduler = CosineAnnealingLR(optimizer, 20, 1e-5)
loss_fn = nn.MSELoss()
kf = KFold(n_splits=5, shuffle=True, random_state=0)
ls = []

for fold, (train_index, val_index) in enumerate(kf.split(X)):
    train_dataset = MyDataset(X[train_index], y[train_index])
    val_dataset = MyDataset(X[val_index], y[val_index])
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    for epoch in range(500):
        train(model, train_loader, optimizer)
        val_loss = evaluate(model, val_loader)
        print("Fold {}, Epoch {}, Val Loss: {:.4f}".format(fold, epoch+1, val_loss))

        if fold == 0 and epoch == 0:
            best_loss = val_loss
        elif val_loss < best_loss:
            best_loss = val_loss
            filename = 'model_best_{}-{}.pth'.format(fold, epoch+1)
            torch.save(model.state_dict(), filename)
            print("********************************BEST {}-RMSE: {:.2e}*****************************".format(fold, best_loss))

        if epoch == 499:
            ls.append(best_loss)

# Save the loss values to a CSV file
p = pd.DataFrame(ls)
filename = 'loss.csv'
p.to_csv(filename)


# Plot the loss values
#
