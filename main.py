import os
import numpy as np
import datetime
import torch
from torch.profiler import ProfilerActivity
from torch.autograd import Variable

ct = datetime.datetime.now()
tensorboard_dir = "./logs"
os.makedirs(tensorboard_dir, exist_ok=True)
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

class Model(torch.nn.Module):
    def __init__(self, inp_size, out_size):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(inp_size, out_size)
    
    def forward(self, x):
        return self.linear(x)
    
inp_size = 1024
batchsize = 32
epochs = 100
lr = 0.01

model = Model(inp_size, 1)
##### For GPU #######
if torch.cuda.is_available():
    model.cuda()


criterion = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Data
# create dummy data for training
x_values = [[i] * inp_size for i in range(batchsize)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, inp_size)

y_values = [2*i + 1 for i in range(batchsize)]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

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
        loss.backward()

        # update parameters
        optimizer.step()

        print('epoch {}, loss {}'.format(epoch, loss.item()))
        
        if hasattr(prof, 'step'):
            prof.step()