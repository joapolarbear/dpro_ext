import os
import json
from contextlib import nullcontext
import datetime

from utils.metadata import RuntimeInfo

import torch
from torch.profiler import ProfilerActivity

class Recorder:
    def __init__(self, models, names, verbose=0, save="./logs", use_profiler=True):
        self.verbose = verbose
        self.save = save

        self.target_module = []
        self.hook_handlers = []
        self.fullname_ops = []
        
        if not isinstance(models, (list, tuple)):
            models = [models]
        if not isinstance(names, (list, tuple)):
            names = [names]
        assert len(models) == len(names)
        for model, name in zip(models, names):
            self.recur_register_hook(model, name)
        
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
                x.dumps()
                
        if self.save is not None:
            if not os.path.exists(self.save):
                os.makedirs(self.save, exist_ok=True)
            
            save_path = os.path.join(self.save, "metadata.json")
            with open(save_path, 'w') as fp:
                json.dump({"fullname_order": [list(runtime_info) for 
                    runtime_info in self.fullname_ops]}, fp,
                    # indent=4
                    )
        
    def make_hook_fn(self, name):
        def hook_fn(module, input, output):
            runtime_info = RuntimeInfo(
                full_name=name, 
                module_name=str(type(module)),
                para_shapes=[list(p.shape) for p in module.parameters()],
                para_dtypes=[str(p.dtype) for p in module.parameters()],
                para_requires_grad=[p.requires_grad for p in module.parameters()])
            self.fullname_ops.append(runtime_info)
            # print(name, output.shape, output.grad_fn)
            # for inp in input:
            #     print(inp.grad_fn)
        return hook_fn

    def _reg_hooks_for_submodules(self, module, name):
        # Register hooks for submodules
        sub_modules = module.__dict__['_modules']
        for sub_name, sub_module in sub_modules.items():
            if sub_module is None:
                continue
            sub_module_full_name = f"{name}/{sub_name}" if len(name) > 0 else sub_name
            self.recur_register_hook(sub_module, sub_module_full_name)
    
    def _reg_hook_implt(self, module, name):
        handler = module.register_forward_hook(self.make_hook_fn(name))
        self.hook_handlers.append(handler)
            
    def recur_register_hook(self, module, name):
        if isinstance(module, (torch.nn.Container,
                torch.nn.Sequential)) or \
                str(type(module)).startswith("<class 'torchvision.models"):
            self._reg_hooks_for_submodules(module, name)
        elif str(type(module)).startswith("<class 'torch.nn") or \
                type(module) in self.target_module:
            self._reg_hook_implt(module, name)
        else:
            inp = input(f"Encounter module {type(module)}, do you want to more fine-grained traces?[y/N]: ")
            if inp.lower() in ["1", "y", "yes"]:
                self._reg_hooks_for_submodules(module, name)
            else:
                self.target_module.append(type(module))
                self._reg_hook_implt(module, name)
    
