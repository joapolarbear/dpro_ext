import os
import re
import json


class TraceTreeNode:
    def __init__(self, e_id, event, sorted_e_id, trace_tree):
        self.e_id = e_id
        self.sorted_e_id = sorted_e_id
        self.event = event
        self.l = event["ts"]
        self.r = event["ts"] + event["dur"]
        
        self.trace_tree = trace_tree
        
        self.childs = []
        self.parent = None
        
        # Flow ids that starts from this event or its child events
        if "in_flow_ids" in event["args"]:
            self.in_flow_ids = set(event["args"]["in_flow_ids"])
        else:
            self.in_flow_ids = set()
        
        # Flow ids that ends with this event
        if "out_flow_ids" in event["args"]:
            self.out_flow_ids = set(event["args"]["out_flow_ids"])
        else:
            self.out_flow_ids = set()
        
        # For leaf node with cuda event, cuda_ts and cuda_dur correspond to the cuda event
        # For non-leaf node which invokes cuda operations, cuda_ts and cuda_dur correspond to
        # the cuda events of the child leaf nodes
        self.cuda_event = None
        self.cuda_ts = None
        self.cuda_dur = None
        if event["cat"] in ["kernel", "gpu_memcpy"]:
            self.cuda_ts = event["ts"]
            self.cuda_dur = event["dur"]
            
        in_ac2g_flow_ids = [flow_id for flow_id in self.in_flow_ids if self.trace_tree.flow_keyed_by_id[flow_id].is_cpu2gpu]
        assert len(in_ac2g_flow_ids) <= 1
        if len(in_ac2g_flow_ids) == 1:
            # CPU to GPU dependency
            flow = self.trace_tree.flow_keyed_by_id[in_ac2g_flow_ids[0]]
            gpu_e_id = flow.out_e_id
            if gpu_e_id is None:
                print(f"Warning: the following CPU event has no corresponding GPU event: {self.trace_tree.torch_traces[flow.in_e_id]}")
            else:
                self.cuda_event = self.trace_tree.torch_traces[gpu_e_id]
                self.cuda_ts = self.cuda_event["ts"]
                self.cuda_dur = self.cuda_event["dur"]
                self.cuda_dur = self.cuda_event["dur"]

    @property
    def out_flow_id(self):
        assert len(self.out_flow_ids) == 1
        return list(self.out_flow_ids)[0]
            
    def add_child(self, node):
        self.childs.append(node)
        node.parent = self
        self.align_flow_info_w_child(node)
    
    def align_flow_info_w_child(self, child):
        # Should be called when traversing the trace tree in post order
        self.in_flow_ids = self.in_flow_ids.union(child.in_flow_ids)
        # NOTE: do not align out flow IDs, since we want to know 
        # the exact GPU event ID or BWD event ID
        # self.out_flow_ids = self.out_flow_ids.union(child.out_flow_ids)
        
        if child.cuda_ts is not None:
            if self.cuda_ts is None:
                self.cuda_ts = child.cuda_ts
                self.cuda_dur = child.cuda_dur
            else:
                self.cuda_dur = child.cuda_ts + child.cuda_dur - self.cuda_ts
   

class ProfileLevelDecider:
    # Decide the profile level
    def __init__(self):
        pass
    
    def make_decision(self, node: TraceTreeNode, parents):
        pass
    
    def accept_decision(self, node: TraceTreeNode, decision: str):
        pass
    
    def withdraw_decision(self, node: TraceTreeNode):
        pass
    
class ManualProfileLevelDecider(ProfileLevelDecider):
    # Manually decide the profile level in a interactive manner
    def __init__(self):
        super().__init__()
        self.name2decision = {}
        self.name2decision_cfg_path = "name2decision_cfg.json"
        if os.path.exists(self.name2decision_cfg_path):
            with open(self.name2decision_cfg_path, 'r') as fp:
                self.name2decision = json.load(fp)
            print(f"Load name2decision cfg file from {self.name2decision_cfg_path}")
        
    def make_decision(self, node: TraceTreeNode, parents):
        if node.cuda_ts is None:
            # Breakdown those nodes by default
            print(f"node cuda_ts is None {node.event['name']=}")
            return 'p'
        
        if node.event["name"] in self.name2decision:
            # We have seen events with the same, directly make the decision
            decision = self.name2decision[node.event["name"]]
            # print(f"Re-use the decsion {decision} for node {node.event['name'][:200]}")
            return decision
            
        while True:
            print("\n")
            print("=" * 100)
            for level, p_node in enumerate(parents):
                print("    " * level + p_node.event["name"][:200])
            print("    " * (len(parents)) + f"{node.event['name'][:200]} <-------------------- Cur Node")
            # print("    " * (len(parents)+1) + "childs:")
            for child in node.childs[:12]:
                print("    " * (len(parents)+1) + f"{child.event['name'][:200]}")
            if len(node.childs) > 12:
                for _ in range(3):
                    print("    " * (len(parents)+1) + f".")
            print("-" * 100)
            print(f" s\t\tShow full name of the node")
            print(f" g\t\tGenerate a event, type `g <customized name>` to change the trace name")
            print(f" b\t\t(default) Breakdown")
            print(f" u\t\tUP and undo")
            print(f" k\t\tKeep all the CPU and GPU events related to this node and all its child, `k <dir>`")
            print(f" p\t\tPass this node, do nothing")
            print(" Note: By default, the same decision will be used for the same event, add * after command disable this")
            inp = input("Please input commands: ")
            print("=" * 100)
            # Default operation
            if len(inp) == 0:
                inp = "b"
            if inp == 's':
                print(f"The full name is: {node.event['name']}")
            elif inp == "b" and len(node.childs) == 0:
                print(f"\n!!! Do not allow breakdowning `{node.event['name']}` which has no child")
            elif inp[0] not in ["b", "p", "g", "k", "u"]:
                print(f"\n!!! Invalid selection: {inp}")
            elif inp == "u":
                # Withdraw, re-process the parent node
                decision = inp
                break
            elif inp[0] == "p":
                if read_yes(f"Do you want to skip node `{node.event['name']}`?"):
                    decision = inp
                    break
            else:
                # Decision = 'b' or 'g' or 'k'
                decision = inp
                break
        return decision

    def accept_decision(self, node: TraceTreeNode, decision: str):
        node_event_name = node.event["name"]
        self.name2decision[node_event_name] = decision
    
    def withdraw_decision(self, node: TraceTreeNode):
        node_event_name = node.event["name"]
        if node_event_name in self.name2decision:
            self.name2decision.pop(node_event_name)
    
    def dump(self):
        with open(self.name2decision_cfg_path, 'w') as fp:
            json.dump(self.name2decision, fp, indent=4)

class AutoProfileLevelDecider(ProfileLevelDecider):
    def __init__(self, metadata):
        super().__init__()
        
        # list of (fullname, class_name)
        self.fullname_order = [(full_name, self.process_class_name(class_name)) for 
                               (full_name, class_name) in metadata["fullname_order"]]
        self.ptr = 0
        
    def process_class_name(self, class_name):
        rst = re.search(r"<class '(?P<module_name>[\w\.]+)'>", class_name)
        assert rst is not None, (class_name)
        return rst["module_name"].split(".")[-1]
    
    def make_decision(self, node: TraceTreeNode, parents):
        if self.ptr < len(self.fullname_order) and self.fullname_order[self.ptr][1] in node.event["name"]:
            fullname = self.fullname_order[self.ptr][0]
            self.ptr += 1

            # Register the full module name of the flow if exists
            in_fwd2bwd_flow_ids = [flow_id for flow_id in node.in_flow_ids if node.trace_tree.flow_keyed_by_id[flow_id].is_fwd2bwd]
            if len(in_fwd2bwd_flow_ids) > 0:
                for idx, flow_id in enumerate(in_fwd2bwd_flow_ids): 
                    node.trace_tree.flow_keyed_by_id[flow_id].register_fullname(f"{fullname}_sub{idx}")
                
            return f"g {fullname}"
        elif len(node.out_flow_ids) > 0:
            out_flow = node.trace_tree.flow_keyed_by_id[node.out_flow_id]
            assert out_flow.is_fwd2bwd
            if out_flow.fullname is None:
                if node.cuda_ts is None:
                    # TODO (huhanpeng): ignore those CPU only ops, like aten::view,
                    pass
                else:
                    import pdb; pdb.set_trace()
            return f"g backward/{out_flow.fullname}"
        elif node.event["name"].startswith("aten::"):
            return "g"
        else:
            return "b"


FLOW_CNT = 0

def get_flow_cnt():
    global FLOW_CNT
    FLOW_CNT += 1
    return FLOW_CNT - 1

def read_yes(prompt):
    inp = input(f"{prompt} (Y/n): ")
    if inp.lower() in ['', 'y', 'yes', '1']:
        return True
    else:
        return False
     
'''
The profiler allow users to specify the granularity for trace recording
when analyzing the raw output of torch's Profiler in an offline manner. 
E.g., given a torch module A containing op B, another module C, current 
profilers supports to record the CUDA runtime corresponding to 
1) module A; 
2) op B and module C. 
The reason to to this is that with Torch, users usually implement their 
own modules and one module may invoke another module. Torch's Profiler 
will record the call stack, from the highest level - the module represe
nting the DNN model - to the lowest level - CPU event to launch a CUDA 
kernel. It's hard to decide which modules and operators users may be 
intrested in when analzing the traces.
'''
class TraceTree:
    def __init__(self, torch_traces, metadata, pid2info, flow_keyed_by_id):
        self.torch_traces = torch_traces
        self.metadata = metadata
        self.pid2info = pid2info
        self.flow_keyed_by_id = flow_keyed_by_id
        
        self.root_nodes = []
        self.node_ptr = None
        
        # self.pld = ManualProfileLevelDecider()
        self.pld = AutoProfileLevelDecider(metadata)
    
    def add_event(self, e_id, new, sorted_e_id):
        node = TraceTreeNode(e_id, new, sorted_e_id, self)
        succ = True
        if len(self.root_nodes) == 0:
            self.root_nodes = [node]
        else:
            assert self.node_ptr.l <= node.l, (node.event["name"][:100], self.node_ptr.event["name"][:100])
            succ = self._recur_add_node(self.node_ptr, node)
        if succ:
            self.node_ptr = node
        
    def _recur_add_node(self, prev, new):
        ''' If a node is added, return True, otherwise, return False
        '''
        if new.l >= prev.r:
            # The new node is at the same level as
            # prev:   l       r
            # new:              l      r
            if prev.parent is None:
                # Root nodes
                self.root_nodes.append(new)
                return True
            else:
                prev.parent.align_flow_info_w_child(prev)
                return self._recur_add_node(prev.parent, new)
        elif new.r <= prev.r:
            # The new node is the child of the current node
            # prev:   l       r
            # new:      l   r
            prev.add_child(new)
            return True
        else:
            # The following case is not allowed in the same thread
            # prev:   l       r
            # new:      l        r
            print(prev.event)
            print(new.event)
            raise RuntimeError()
            return False
    
    def show(self):
        level = 0
        for node in self.root_nodes:  
            self._recur_show(node, level)
    
    def _recur_show(self, node, level):
        raise NotImplementedError()
        msg = "\t" * level
        if node.event["cat"] == "cpu_op":
            msg += "* "
        msg += node.event["name"][:100]
        if node.cuda_ts:
            msg += f" ({node.cuda_ts - min_ts} - {node.cuda_ts + node.cuda_dur - min_ts})"
        if node.e_id in flow_event_to_event:
            cuda_event = traces[flow_event_to_event[node.e_id]]
            msg += f" --> {cuda_event['name'][:100]}"
        print(msg)
        for child in node.childs: 
            self._recur_show(child, level+1)
    
    def gen_traces(self):
        self.trace_name_cnt = {}
        self.rst_traces = []
        print("\n########### Interactive generate events #########")
        for node in self.root_nodes:
            # Loop support withdraw operations
            while True:
                events, proc_next_child = self._recur_gen_traces(node, [])
                if proc_next_child:
                    self.rst_traces.extend(events)
                    break
        return self.rst_traces

    def _recur_gen_traces(self, node, parents):
        '''
        Return
        events: list of event
            The traces corresponding to this node and it's child nodes
        proc_next_child: bool
            Set True to process the sibling of this node, i.e., the next child of the parent of this node 
        '''
        match = re.search(r"(?P<module_name>nn.Module:.*)_\d+", node.event["name"])
        if match is not None:
            node.event["name"] = match.groupdict()["module_name"]
        match = re.search(r"(?P<module_name>autograd::engine::evaluate_function:.*)\d+", node.event["name"])
        if match is not None:
            node.event["name"] = match.groupdict()["module_name"]
        while True:
            decision = self.pld.make_decision(node, parents)
            if decision == "u":
                # Withdraw, re-process the parent node
                return None, False
            
            if decision.endswith("*"):
                decision = decision.split("*")[0]
            else:
                self.pld.accept_decision(node, decision)
                
            if decision.startswith("g"):
                # Generate a event for this node
                event = self.gen_cpu_event_with_cuda_ts(node)
                if decision.startswith("g ") and len(decision) > 2:
                    self.pld.withdraw_decision(node)
                    event["name"] = decision.split("g ")[1]
                return [event], True
            elif decision == "b":
                # break down this node
                re_proc_this_node = False
                ret_events = []
                for child_id, child in enumerate(node.childs):
                    events, proc_next_child = self._recur_gen_traces(child, parents + [node])
                    
                    # Double confirm the withdraw operation
                    if not proc_next_child and len(ret_events) > 0:
                        if read_yes(f"Do you want to redo `{node.event['name']}`, whose {child_id} childs have been processed with {len(ret_events)} events"):
                            pass
                        else:
                            proc_next_child = True
                            
                    if proc_next_child:
                        ret_events.extend(events)
                    else:
                        re_proc_this_node = True
                        self.pld.withdraw_decision(node)
                        break
                if not re_proc_this_node:
                    return ret_events, True
            elif decision.startswith("k"):
                # Keep all traces under this node, if there is an argument like
                # `k <path>`, this part of traces will be saved in `<path>` separately
                sorted_e_id = node.sorted_e_id
                events = self.keep_all_traces(node)
                if decision.startswith("k ") and len(decision) > 2:
                    dump_dir = decision.split("k ")[1]
                    dump_node_name, _ = self.remove_invoking_file_path(node.event["name"])
                    dump_node_name = dump_node_name.replace(": ", "_").replace(" ", "_").replace(".", "_")
                    path = os.path.join(dump_dir, f"{dump_node_name}.json")
                    repeat_file_cnt = len([f for f in os.listdir(dump_dir) if re.match(f"{dump_node_name}\d*\.json", f)])
                    path = os.path.join(dump_dir, f"{dump_node_name}{repeat_file_cnt}.json")
                    with open(path, 'w') as fp:
                        json.dump({"traceEvents": events}, fp)
                    events = []
                return events, True
            elif decision == "p":
                # Pass this node and its child nodes
                return [], True
        raise RuntimeError("Not expected")
    
    def keep_all_traces(self, node):
        # Keep all the CPU and GPU events related to this node and all its child
        events = [node.event]
        if node.cuda_event:
            flow_cnt = get_flow_cnt()
            events.append({
                "ph": "s", "id": flow_cnt, "pid": node.event["pid"], 
                "tid": node.event["tid"], "ts": node.event["ts"],
                "cat": "ac2g", "name": "ac2g"
            })
            events.append(node.cuda_event)
            events.append({
                "ph": "f", "id": flow_cnt, "pid": node.cuda_event["pid"], 
                "tid": node.cuda_event["tid"], "ts": node.cuda_event["ts"],
                "cat": "ac2g", "name": "ac2g", "bp": "e"
            })
        for child in node.childs:
            child_events = self.keep_all_traces(child)
            events.extend(child_events)
        return events
        
    def remove_invoking_file_path(self, event_name):
        name_prefix = None
        match = re.search(r"(?P<name_prefix>.*\.py\(\d+\): )(?P<module_name>.*)", event_name)
        if match is not None:
            match_rst = match.groupdict()
            event_name = match_rst["module_name"]
            name_prefix = match_rst["name_prefix"]
        return event_name, name_prefix
        
    def gen_cpu_event_with_cuda_ts(self, node):
        name_prefix = None
        match = re.search(r"nn.Module: (?P<module_name>.*)", node.event["name"])
        if match is not None:
            node.event["name"] = match.groupdict()["module_name"]
            name_prefix = "nn.Module: "
        match = re.search(r"autograd::engine::evaluate_function: (?P<module_name>.*)", node.event["name"])
        if match is not None:
            node.event["name"] = match.groupdict()["module_name"]
            name_prefix = "autograd::engine::evaluate_function: "
        
        node.event["name"], name_prefix = self.remove_invoking_file_path(node.event["name"])
            
        if node.event["name"] not in self.trace_name_cnt:
            self.trace_name_cnt[node.event["name"]] = 0
        else:
            self.trace_name_cnt[node.event["name"]] += 1
        
        pid = self.pid2info[node.event["pid"]]["process_name"] + " " \
                + self.pid2info[node.event["pid"]]["process_labels"]
        tid = self.pid2info[node.event["pid"]]["threads"][node.event["tid"]]
        if node.event["name"] == "isend":
            tid += " SEND"
        elif node.event["name"] == "irecv":
            tid += " RECV"
        
        event = {
            "name": node.event["name"],
            "ph": "X",
            "ts": node.cuda_ts,
            "dur": node.cuda_dur,
            "cat": "cuda",
            "pid": pid,
            "tid": tid,
            "args": {
                "name": f'{node.event["name"]}_{self.trace_name_cnt[node.event["name"]]}'
            }
        }
        if name_prefix is not None:
            event["args"]["name_prefix"] = name_prefix
        return event