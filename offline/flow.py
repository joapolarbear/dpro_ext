
class Flow:
    def __init__(self, torch_traces, id, cat, in_e_id=None, out_e_id=None):
        self.torch_traces = torch_traces
        self.id = id
        self.cat = cat
        self.in_e_id = in_e_id
        self.out_e_id = out_e_id
        # Full module name, only for fwd2bwd flows
        self.fullname = None
        
    @property
    def is_cpu2gpu(self):
        return self.cat == "ac2g"

    @property
    def is_fwd2bwd(self):
        return self.cat == "fwdbwd"
        
    def register_in_event(self, in_e_id: int):
        # Register an event that starts a flow
        self.in_e_id = in_e_id
        if "in_flow_ids" not in self.torch_traces[in_e_id]["args"]:
            self.torch_traces[in_e_id]["args"]["in_flow_ids"] = [self.id]
        else:
            self.torch_traces[in_e_id]["args"]["in_flow_ids"].append(self.id)
    
    def register_out_event(self, out_e_id: int):
        # Register an event that ends a flow
        self.out_e_id = out_e_id
        if "out_flow_ids" not in self.torch_traces[out_e_id]["args"]:
            self.torch_traces[out_e_id]["args"]["out_flow_ids"] = [self.id]
        else:
            self.torch_traces[out_e_id]["args"]["out_flow_ids"].append(self.id)
            
    def register_fullname(self, fullname):
        self.fullname = fullname
        
       