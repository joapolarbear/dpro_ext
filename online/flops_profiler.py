from loguru import logger
from flops_profiler.profiler import FlopsProfiler, _flops_to_string, _params_to_string

class WrapFlopsProfiler:
    def __init__(self, models, *args, prefix="", **kwargs):
        if not isinstance(models, list):
            models = [models]
        self.profs = [FlopsProfiler(model, *args, **kwargs) for model in models]
        self.profile_iter = 3
        self.prefix = prefix
        
    def __enter__(self):
        self.iter_num = 0
        return self

    def __exit__(self, *args, **kwargs):
        self.iter_num = None
    
    def step(self):
        self.iter_num += 1
        if self.iter_num == self.profile_iter:
            for prof in self.profs:
                prof.start_profile()
        
        if self.iter_num == self.profile_iter + 1:
            total_flops = 0
            total_params = 0
            for prof in self.profs:
                prof.stop_profile()
                total_flops += prof.get_total_flops(as_string=False)
                total_params += prof.get_total_params(as_string=False)
                prof.end_profile()
            total_flops = _flops_to_string(total_flops)
            total_params = _params_to_string(total_params)
            print(f"{self.prefix} {total_flops=}, {total_params=}")
            