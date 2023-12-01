from collections import namedtuple

RuntimeInfoCls = namedtuple("RuntimeInfo", [
    "full_name",
    "module_name",
    "para_shapes",
    "para_dtypes",
    "para_requires_grad"
])

class RuntimeInfo(RuntimeInfoCls):
    def dumps(self):
        print(f"{self.full_name=}, {self.module_name=}, {self.para_shapes=}, {self.para_dtypes=}, {self.para_requires_grad=}")
    
