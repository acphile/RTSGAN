import torch
import torch.nn as nn
from torch.nn import functional as F
from basic import PositionwiseFeedForward

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(Generator, self).__init__()
        def block(inp, out):
            return nn.Sequential(
                nn.Linear(inp, out),
                nn.LayerNorm(out),
                nn.LeakyReLU(0.2),
            )

        self.block_0 = block(input_dim, input_dim)
        self.block_1 = block(input_dim, input_dim)
        self.block_2 = block(input_dim, input_dim)
        self.block_4 = nn.Linear(input_dim, hidden_dim*(layers+1))
        self.final = nn.LeakyReLU(0.2)
        self.block_3 = block(input_dim, input_dim)

    def forward(self, x):
        x1 = self.block_0(x) + x
        x1 = self.block_1(x1) + x1
        x1 = self.block_2(x1) + x1
        x1 = self.block_3(x1) + x1
        return self.final(self.block_4(x1))

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, (2 * input_dim) // 3),
            nn.LeakyReLU(0.2),
            nn.Linear((2 * input_dim) // 3, input_dim // 3),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim // 3, 1),
        )

    def forward(self, x):
        return self.model(x)



