import os
import numpy as np

import torch
from torch.autograd import Variable

from online.recorder import Recorder

# Model definition
inp_size = 1024
batch_size = 32
epochs = 10
lr = 0.01

if False:
    from model.mlp import MLP, mlp_dummpy_data
    model = MLP(inp_size, 1)
    x_train, y_train = mlp_dummpy_data(inp_size, batch_size)
elif True:
    from model.cnn import resnet50, cnn_dummpy_data
    model = resnet50()
    x_train, y_train = cnn_dummpy_data(batch_size)

criterion = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

recorder = Recorder([model, criterion], ["", "loss"])

##### For GPU #######
if torch.cuda.is_available():
    model.cuda()
    criterion.cuda()

# Converting inputs and labels to Variable
if torch.cuda.is_available():
    inputs = Variable(torch.from_numpy(x_train).cuda())
    labels = Variable(torch.from_numpy(y_train).cuda())
else:
    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_train))
    

from online.flops_profiler import WrapFlopsProfiler
ds_engine = False
flops_prof = WrapFlopsProfiler(model, ds_engine if ds_engine else None)


# Training
with flops_prof as fprof:  
    with recorder as prof:
        for epoch in range(epochs):
            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
            optimizer.zero_grad()
            
            # get output from the model, given the inputs
            outputs = model(inputs)

            # get loss for the predicted output
            loss = criterion(outputs, labels)
            # get gradients w.r.t to parameters
            loss.backward(retain_graph=False)

            # update parameters
            optimizer.step()

            print('epoch {}, loss {}'.format(epoch, loss.item()))
            
            prof.step()
            fprof.step()
            
recorder.summary()