import numpy as np
import torch

class MLP(torch.nn.Module):
    def __init__(self, inp_size, out_size):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(inp_size, 128, bias=False)
        self.linear2 = torch.nn.Linear(128, out_size, bias=False)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

# Data
def mlp_dummpy_data(inp_size, batch_size):
    # create dummy data for training
    x_values = [[i] * inp_size for i in range(batch_size)]
    x_train = np.array(x_values, dtype=np.float32)
    x_train = x_train.reshape(-1, inp_size)

    y_values = [2*i + 1 for i in range(batch_size)]
    y_train = np.array(y_values, dtype=np.float32)
    y_train = y_train.reshape(-1, 1)
    return x_train, y_train
