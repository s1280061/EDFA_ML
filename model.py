import torch
from torch import nn

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(85, 256),
            nn.BatchNorm1d(256),    # 各バッチの中の各特徴量が平均0, 標準偏差(散らばり度合)1に正規化
            nn.ReLU(),  # 入力が0未満の場合は0、それ以外の場合は入力をそのまま出力する関数
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(128, 83)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)      # 線形変換
        y = self.decoder(x)
        return y





