import os
import json
from contextlib import nullcontext
import datetime

import torch
from torch.profiler import ProfilerActivity

class Recorder:
    def __init__(self, model, verbose=0, save="./logs", use_profiler=True):
        self.verbose = verbose
        self.save = save

        self.target_module = []
        self.hook_handlers = []
        self.fullname_ops = []
        
        self.recur_register_hook(model)
        
        if use_profiler:
            tensorboard_dir = "./logs"
            os.makedirs(tensorboard_dir, exist_ok=True)
            ct = datetime.datetime.now()
            self.profiler = torch.profiler.profile(
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
            self.profiler = nullcontext()
        self.enter_handler = None
        
    def clear_hook(self):
        for handler in self.hook_handlers:
            handler.remove()
    
    def step(self):
        self.clear_hook()
        if hasattr(self.enter_handler, 'step'):
            self.enter_handler.step()
            
    def __enter__(self):
        self.enter_handler = self.profiler.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        self.profiler.__exit__(*args, **kwargs)
        self.enter_handler = None
        
    def summary(self):
        if self.verbose > 0:
            for x in self.fullname_ops:
                print(x)
                
        if self.save is not None:
            if not os.path.exists(self.save):
                os.makedirs(self.save, exist_ok=True)
            
            save_path = os.path.join(self.save, "metadata.json")
            with open(save_path, 'w') as fp:
                json.dump({"fullname_order": self.fullname_ops}, fp, indent=4)
        
    def make_hook_fn(self, name):
        def hook_fn(module, input, output):
            self.fullname_ops.append((name, str(type(module))))
            # print(name, output.shape, output.grad_fn)
            # for inp in input:
            #     print(inp.grad_fn)
        return hook_fn

    def recur_register_hook(self, module, prefix=""):
        sub_modules = module.__dict__['_modules']
        for name, sub_module in sub_modules.items():
            sub_module_full_name = f"{prefix}/{name}" if len(prefix) > 0 else name
            if sub_module is None or isinstance(sub_module, torch.nn.Module) is False:
                continue
            if isinstance(sub_module, torch.nn.Container) or \
                    isinstance(sub_module, torch.nn.Sequential) or \
                    str(type(sub_module)).startswith("<class 'torchvision.models"):
                self.recur_register_hook(sub_module, prefix=sub_module_full_name)
            elif str(type(sub_module)).startswith("<class 'torch.nn.modules") or \
                    type(sub_module) in self.target_module:
                handler = sub_module.register_forward_hook(self.make_hook_fn(sub_module_full_name))
                self.hook_handlers.append(handler)
            else:
                inp = input(f"Encounter module {type(sub_module)}, do you want to more fine-grained traces?[y/N]: ")
                if inp.lower() in ["1", "y", "yes"]:
                    self.recur_register_hook(sub_module, prefix=sub_module_full_name)
                else:
                    self.target_module.append(type(sub_module))
                    handler = sub_module.register_forward_hook(self.make_hook_fn(sub_module_full_name))
                    self.hook_handlers.append(handler)
