import os, sys
import json
from functools import cmp_to_key
import re

log_dir = sys.argv[1]

def load_torch_profiler_rst(torch_profiler_path):
    print(f"Torch Profiler Trace loading at {torch_profiler_path[:50]} ... ")
    with open(torch_profiler_path, 'r') as fp:
        traces = json.load(fp)
    return traces

def load(log_dir):
    fullname_order_path = os.path.join(log_dir, "metadata.json")
    if not os.path.exists(fullname_order_path):
        print(f"Metadata is not available under the directory {log_dir}")
    with open(fullname_order_path, 'r') as fp:
        metadata = json.load(fp)
        
    _root, _dirs, _files = list(os.walk(log_dir))[0]
    if len(_dirs) > 0:
        _dirs = sorted([d for d in _dirs if re.match(r"\d{8}-\d{6}", d)])
        torch_profiler_dir = os.path.join(_root, _dirs[-1])
        print(f"There are multiple torch profiler traces, choose the latest one: {torch_profiler_dir}")
    else:
        torch_profiler_dir = os.path.join(_root, _dirs[0])
    files = os.listdir(torch_profiler_dir)
    assert len(files) == 1
    torch_profiler_path = os.path.join(torch_profiler_dir, (files[0]))
    torch_traces = load_torch_profiler_rst(torch_profiler_path)
    return metadata, torch_traces

metadata, torch_traces = load(log_dir)

def extract_pid_and_flow_info(traces):
    pid2info = {}
    flow_id_to_events = {}
    print("Stat pid info ... ")
    for e_id, event in enumerate(traces["traceEvents"]):
        if event["ph"] == 'M':
            if event["pid"] not in pid2info:
                pid2info[event["pid"]] = {"threads": {}}
            if event["name"] == "process_name":
                pid2info[event["pid"]]["process_name"] = event["args"]["name"]
            elif event["name"] == "process_labels":
                pid2info[event["pid"]]["process_labels"] = event["args"]["labels"]
            elif event["name"] == "thread_name":
                pid2info[event["pid"]]["threads"][event["tid"]] = event["args"]["name"]
        elif event["ph"] == "s":
            source_event = traces["traceEvents"][e_id-1]
            assert source_event["pid"] == event["pid"]
            assert source_event["tid"] == event["tid"]
            assert source_event["ts"] == event["ts"]
            # TODO what's the usage of the `cat` and `name` fields for this flow evnet
            if event["id"] not in flow_id_to_events:
                flow_id_to_events[event["id"]] = [None, None, event["cat"]]
            flow_id_to_events[event["id"]][0] = e_id-1
        elif event["ph"] == "f":
            target_event = traces["traceEvents"][e_id-1]
            assert target_event["pid"] == event["pid"]
            assert target_event["tid"] == event["tid"]
            assert target_event["ts"] == event["ts"]
            if event["id"] not in flow_id_to_events:
                flow_id_to_events[event["id"]] = [None, None, event["cat"]]
            flow_id_to_events[event["id"]][1] = e_id-1
            
    flow_event_to_event = {}
    for _id, (se, te, cat) in flow_id_to_events.items():
        # TODO (huhanpeng): need to use the following assertation
        assert se is not None, (id)
        assert te is not None, (id)
        if se is None or te is None:
            continue
        # print(se["name"][:100], "  ->  ", te["name"][:100])
        flow_event_to_event[se] = (te, cat)
    return pid2info, flow_event_to_event
    
pid2info, flow_event_to_event = extract_pid_and_flow_info(torch_traces)

def event_cmp_func(o1, o2):
    e_id1, e1 = o1
    e_id2, e2 = o2
    if e1["pid"] != e2["pid"]:
        return int(e1["pid"]) - int(e2["pid"])
    elif e1["tid"] != e2["tid"]:
        return int(e1["tid"]) - int(e2["tid"])
    else:
        return e1["ts"] - e2["ts"]
    
sorted_traces = sorted(
    [(e_id, e) for e_id, e in enumerate(torch_traces["traceEvents"]) if 
        (isinstance(e["pid"], int) or e["pid"].isdigit()) and e["ph"] == "X"],
    key=cmp_to_key(event_cmp_func))
min_ts = sorted_traces[0]["ts"]


def read_yes(prompt):
    inp = input(f"{prompt} (Y/n): ")
    if inp.lower() in ['', 'y', 'yes', '1']:
        return True
    else:
        return False
    
class TraceTreeNode:
    def __init__(self, e_id, event, sorted_e_id):
        self.e_id = e_id
        self.sorted_e_id = sorted_e_id
        self.event = event
        self.l = event["ts"]
        self.r = event["ts"] + event["dur"]
        
        self.childs = []
        self.parent = None
        
        # For leaf node with cuda event, cuda_ts and cuda_dur correspond to the cuda event
        # For non-leaf node which invokes cuda operations, cuda_ts and cuda_dur correspond to
        # the cuda events of the child leaf nodes
        if event["cat"] in ["kernel", "gpu_memcpy"]:
            self.cuda_ts = event["ts"]
            self.cuda_dur = event["dur"]
        
        if e_id in flow_event_to_event:
            te_id, cat = flow_event_to_event[e_id]
            if cat == "ac2g":
                # CPU to GPU dependency
                self.cuda_event = torch_traces["traceEvents"][te_id]
                self.cuda_ts = self.cuda_event["ts"]
                self.cuda_dur = self.cuda_event["dur"]
            elif cat == "fwdbwd":
                # fwd to bwd dependency
                raise 
            else:
                raise ValueError(f"Invalid flow cat {cat}")
        else:
            self.cuda_event = None
            self.cuda_ts = None
            self.cuda_dur = None
        
    def add_child(self, node):
        self.childs.append(node)
        node.parent = self
        
        self.align_cuda_time_w_child(node)
    
    def align_cuda_time_w_child(self, child):
        if child.cuda_ts is not None:
            if self.cuda_ts is None:
                self.cuda_ts = child.cuda_ts
                self.cuda_dur = child.cuda_dur
            else:
                self.cuda_dur = child.cuda_ts + child.cuda_dur - self.cuda_ts
    
FLOW_CNT = 0
def get_flow_cnt():
    global FLOW_CNT
    FLOW_CNT += 1
    return FLOW_CNT - 1

class ProfileLevelDecider:
    def __init__(self):
        pass
    
class ManualProfileLevelDecider(ProfileLevelDecider):
    def __init__(self):
        super().__init__()
        self.name2decision = {}
        name2decision_cfg_path = "name2decision_cfg.json"
        if os.path.exists(name2decision_cfg_path):
            with open(name2decision_cfg_path, 'r') as fp:
                self.name2decision = json.load(fp)
            print(f"Load name2decision cfg file from {name2decision_cfg_path}")
        
    def make_decision(self, node, parents):
        assert node.cuda_ts is not None
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
            elif inp == "u":
                # Withdraw, re-process the parent node
                return None, False
            elif inp == "b" and len(node.childs) == 0:
                print(f"\n!!! Do not allow breakdowning `{node.event['name']}` which has no child")
            elif inp[0] == "p":
                if read_yes(f"Do you want to skip node `{node.event['name']}`?"):
                    decision = inp
                    break
            elif inp[0] not in ["b", "p", "g", "k"]:
                print(f"\n!!! Invalid selection: {inp}")
            else:
                # Decision = 'b' or 'g' or 'k'
                decision = inp
                break
        return decision

class AutoProfileLevelDecider(ProfileLevelDecider):
    pass



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
    def __init__(self):
        self.root_nodes = []
        self.node_ptr = None
        
        self.pld = ManualProfileLevelDecider()
    
    def add_event(self, e_id, new, sorted_e_id):
        node = TraceTreeNode(e_id, new, sorted_e_id)
        succ = True
        if len(self.root_nodes) == 0:
            self.root_nodes = [node]
        else:
            assert self.node_ptr.l <= node.l, (node.event["name"][:100], self.node_ptr.event["name"][:100])
            succ = self.add_node(self.node_ptr, node)
        if succ:
            self.node_ptr = node
        
    def add_node(self, prev, new):
        ''' If a node is added, return True, otherwise, return False
        '''
        if new.l >= prev.r:
            # The new node is at the same level as
            # self:   l       r
            # new:              l      r
            if prev.parent is None:
                # Root nodes
                self.root_nodes.append(new)
                return True
            else:
                prev.parent.align_cuda_time_w_child(prev)
                return self.add_node(prev.parent, new)
        elif new.r <= prev.r:
            # The new node is the child of the current node
            # self:   l       r
            # new:      l   r
            prev.add_child(new)
            return True
        else:
            # The following case is not allowed in the same thread
            # self:   l       r
            # new:      l        r
            # print(prev.event, new.event)
            return False
    
    def show(self):
        level = 0
        for node in self.root_nodes:  
            self._recur_show(node, level)
    
    def _recur_show(self, node, level):
        msg = "\t" * level
        if node.event["cat"] == "cpu_op":
            msg += "* "
        msg += node.event["name"][:100]
        if node.cuda_ts:
            msg += f" ({node.cuda_ts - min_ts} - {node.cuda_ts + node.cuda_dur - min_ts})"
        if node.e_id in flow_event_to_event:
            cuda_event = traces["traceEvents"][flow_event_to_event[node.e_id]]
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
        if node.cuda_ts is None:
            # node.event["name"].startswith("<built-in method") \
            # or node.event["name"].startswith("<built-in function"):
            ret_next_child = True
            ret_events = []
            # Breakdown those nodes by default
            for child in node.childs:
                events, proc_next_child = self._recur_gen_traces(child, parents + [node])
                if proc_next_child:
                    ret_events.extend(events)
                else:
                    ret_next_child = False
                    break
            return ret_events, ret_next_child
        
        match = re.search(r"(?P<module_name>nn.Module:.*)_\d+", node.event["name"])
        if match is not None:
            node.event["name"] = match.groupdict()["module_name"]
        match = re.search(r"(?P<module_name>autograd::engine::evaluate_function:.*)\d+", node.event["name"])
        if match is not None:
            node.event["name"] = match.groupdict()["module_name"]
        while True:
            decision = self.pld.make_decision(node, parents)
            node_event_name = node.event["name"]
            if decision.endswith("*"):
                decision = decision.split("*")[0]
            else:
                name2decision[node_event_name] = decision
            if decision.startswith("g"):
                event = self.gen_cpu_event_with_cuda_ts(node)
                if decision.startswith("g ") and len(decision) > 2:
                    if node_event_name in name2decision:
                        name2decision.pop(node_event_name)
                    event["name"] = decision.split("g ")[1]
                return [event], True
            elif decision == "b":
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
                        if node_event_name in name2decision:
                            name2decision.pop(node_event_name)
                        break
                if not re_proc_this_node:
                    return ret_events, True
            elif decision.startswith("k"):
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
        
        pid = pid2info[node.event["pid"]]["process_name"] + " " \
                + pid2info[node.event["pid"]]["process_labels"]
        tid = pid2info[node.event["pid"]]["threads"][node.event["tid"]]
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

pid_to_trace_tree = {}
for sorted_e_id, (e_id, event) in enumerate(sorted_traces):
    if event["pid"] not in pid_to_trace_tree:
        pid_to_trace_tree[event["pid"]] = {}
    if event["tid"] not in pid_to_trace_tree[event["pid"]]:
        pid_to_trace_tree[event["pid"]][event["tid"]] = TraceTree()
    if event["ph"] == "X":
        # print(event["pid"], event["tid"], event["ts"], event["name"][:100])
        pid_to_trace_tree[event["pid"]][event["tid"]].add_event(e_id, event, sorted_e_id)

# Show trace trees
# for pid in pid_to_trace_tree.keys():
#     for tid in pid_to_trace_tree[pid].keys():
#         print(f"\n{pid} {tid}")
#         pid_to_trace_tree[pid][tid].show()
        

rst_traces = []
for pid in pid_to_trace_tree.keys():
    if pid2info[pid]["process_labels"] != "CPU":
        continue
    for tid in pid_to_trace_tree[pid].keys():
        rst_traces.extend(pid_to_trace_tree[pid][tid].gen_traces())
        
output_file = "tmp.json"
print(f"Dump {len(rst_traces)} events to {output_file}") 
with open(output_file, 'w') as fp:
    json.dump({
        "traceEvents": rst_traces
    }, fp)


with open(name2decision_cfg_path, 'w') as fp:
    json.dump(name2decision, fp, indent=4)