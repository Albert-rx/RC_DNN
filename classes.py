from torch import nn
from Cube3_class import *
import torch

METRIC = "QTM"
SCRAMBLE_LENGTH = {"QTM": 26, "HTM": 20}[METRIC]

class TrainConfig:
    learning_rate = 1e-3   #原来参数为-3
    batch_size_per_depth = 1000 #1000
    num_train_steps = 10000     #10000
    interval_steps_save = 1000  #1000
    interval_steps_plot = 100
    scramble_length = SCRAMBLE_LENGTH

# inference
class SearchConfig:
    ## The wider the beam search, the more time required, but the shorter the solution.
    beam_width = 2**12   #原来为2**12
    ## This can be any number greater than or equal to the Gods Number.
    max_depth = SCRAMBLE_LENGTH * 2
    eval__ = "logits" # following the paper. You are free to change this value to "logits"


class LinearBlock(nn.Module):
    """
    Linear layer with ReLU and BatchNorm
    """
    def __init__(self, input_prev, embed_dim):
        super(LinearBlock, self).__init__()
        self.fc = nn.Linear(input_prev, embed_dim)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, inputs):
        x = inputs
        x = self.fc(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class ResidualBlock(nn.Module):
    """
    Residual block with two linear layers
    """
    def __init__(self, embed_dim):
        super(ResidualBlock, self).__init__()
        self.linearblock_1 = LinearBlock(embed_dim, embed_dim)
        self.linearblock_2 = LinearBlock(embed_dim, embed_dim)

    def forward(self, inputs):
        x = inputs
        x = self.linearblock_1(x)
        x = self.linearblock_2(x)
        x += inputs # skip-connection
        return x


env = Cube3()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.one_hot = nn.functional.one_hot
        self.Stack = nn.Sequential(
            LinearBlock(324, 5000),
            LinearBlock(5000, 1000),
            ResidualBlock(1000),
            ResidualBlock(1000),
            ResidualBlock(1000),
            ResidualBlock(1000),
        )
        self.Prediction = nn.Linear(1000, len(env.moves))

    def forward(self, inputs):
        x = inputs
        x = self.one_hot(x.to(torch.int64), num_classes=6).to(torch.float).reshape(-1, 324)   #here 加上了x后面的.to(torch.int64)，原来没有
        x = self.Stack(x)
        logits = self.Prediction(x)

        return logits
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# global model
# model = Model().to(device)
