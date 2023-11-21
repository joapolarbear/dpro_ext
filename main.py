import os
import numpy as np
import datetime
from contextlib import nullcontext

import torch
from torch.profiler import ProfilerActivity
from torch.autograd import Variable

from recorder import Recorder

use_profiler = False
if use_profiler:
    tensorboard_dir = "./logs"
    os.makedirs(tensorboard_dir, exist_ok=True)
    ct = datetime.datetime.now()
    profiler = torch.profiler.profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=5, warmup=2, active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(tensorboard_dir, ct.strftime("%Y%m%d-%H%M%S"))),
            record_shapes=False,
            profile_memory=False,
            with_stack=True,
            with_flops=True,
            with_modules=True
        )
else:
    profiler = nullcontext()

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


recorder = Recorder(model)

##### For GPU #######
if torch.cuda.is_available():
    model.cuda()

criterion = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Converting inputs and labels to Variable
if torch.cuda.is_available():
    inputs = Variable(torch.from_numpy(x_train).cuda())
    labels = Variable(torch.from_numpy(y_train).cuda())
else:
    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_train))
    
# Training    
with profiler as prof:
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
        
        if hasattr(prof, 'step'):
            prof.step()
            
recorder.summary()