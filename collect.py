import os, sys
import json
from functools import cmp_to_key
import re

f = sys.argv[1]

print("Trace loading ... ")
with open(f, 'r') as fp:
    traces = json.load(fp)

pid2info = {}
flow_record = {}
pid2trace_tree = {}
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
        if event["id"] not in flow_record:
            flow_record[event["id"]] = [None, None]
        flow_record[event["id"]][0] = e_id-1
    elif event["ph"] == "f":
        target_event = traces["traceEvents"][e_id-1]
        assert target_event["pid"] == event["pid"]
        assert target_event["tid"] == event["tid"]
        assert target_event["ts"] == event["ts"]
        if event["id"] not in flow_record:
            flow_record[event["id"]] = [None, None]
        flow_record[event["id"]][1] = e_id-1

min_ts = traces["traceEvents"][0]["ts"]
flow_mapping = {}
for _id, (se, te) in flow_record.items():
    # assert se is not None, (id)
    # assert te is not None, (id)
    if se is None or te is None:
        continue
    # print(se["name"][:100], "  ->  ", te["name"][:100])
    flow_mapping[se] = te


name2decision = {}

class TraceTreeNode:
    def __init__(self, e_id, event):
        self.e_id = e_id
        self.event = event
        self.l = event["ts"]
        self.r = event["ts"] + event["dur"]
        
        self.childs = []
        self.parent = None
        
        if event["cat"] in ["kernel", "gpu_memcpy"]:
            self.cuda_ts = event["ts"]
            self.cuda_dur = event["dur"]
        elif e_id in flow_mapping:
            cuda_event = traces["traceEvents"][flow_mapping[e_id]]
            self.cuda_ts = cuda_event["ts"]
            self.cuda_dur = cuda_event["dur"]
        else:
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
    

class TraceTree:
    def __init__(self):
        self.root_nodes = []
        self.node_ptr = None
    
    def add_event(self, e_id, new):
        node = TraceTreeNode(e_id, new)
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
        if node.e_id in flow_mapping:
            cuda_event = traces["traceEvents"][flow_mapping[node.e_id]]
            msg += f" --> {cuda_event['name'][:100]}"
        print(msg)
        for child in node.childs: 
            self._recur_show(child, level+1)
    
    def gen_traces(self):
        self.trace_name_cnt = {}
        self.rst_traces = []
        print("\n########### Interactive generate events #########")
        for node in self.root_nodes:
            while True:
                events, next_child = self._recur_gen_traces(node, [])
                if next_child:
                    self.rst_traces.extend(events)
                    break
        return self.rst_traces

    def _recur_gen_traces(self, node, parents):
        if node.cuda_ts is None or node.event["name"].startswith("<built-in method"):
            ret_next_child = True
            ret_events = []
            for child in node.childs:
                events, next_child = self._recur_gen_traces(child, parents + [node])
                if next_child:
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
            # node.cuda_ts is not None
            if node.event["name"] in name2decision:
                # We have seen events with the same, directly make the decision
                decision = name2decision[node.event["name"]]
                # print(f"Re-use the decsion {decision} for node {node.event['name'][:200]}")
            else:
                while True:
                    print()
                    for level, p_node in enumerate(parents):
                        print("    " * level + p_node.event["name"][:200])
                    print("    " * (len(parents)-1) + f" == {node.event['name'][:200]} ==")
                    print(f" 0. Show full name of the node")
                    print(f" 1. Generate a event")
                    print(f" 2. (default) Breakdown")
                    print(f" 3. UP")
                    inp = input("Please input 0/1/2/3:")
                    if len(inp) == 0:
                        inp = "2"
                    if inp == '0':
                        print(f"The full name is: {node.event['name']}")
                    elif inp == "3":
                        return None, False
                    elif inp not in ["1", "2"]:
                        print(f"\n!!! Invalid selection: {inp}")
                    else:
                        decision = inp
                        break
            
            name2decision[node.event["name"]] = decision
            if decision == "1":
                event = self.gen_event_one_node(node)
                return [event], True
            elif decision == "2":
                go_back_to_parent = False
                ret_events = []
                for child in node.childs: 
                    events, next_child = self._recur_gen_traces(child, parents + [node])
                    if next_child:
                        ret_events.extend(events)
                    else:
                        go_back_to_parent = True
                        if node.event["name"] in name2decision:
                            name2decision.pop(node.event["name"])
                        break
                if not go_back_to_parent:
                    return ret_events, True
        raise RuntimeError("Not expected")
                
    def gen_event_one_node(self, node):
        if node.event["name"] not in self.trace_name_cnt:
            self.trace_name_cnt[node.event["name"]] = 0
        else:
            self.trace_name_cnt[node.event["name"]] += 1
            
        return {
            "name": node.event["name"],
            "ph": "X",
            "ts": node.cuda_ts,
            "dur": node.cuda_dur,
            "cat": "cuda",
            "pid": pid2info[node.event["pid"]]["process_name"] + " " \
                + pid2info[node.event["pid"]]["process_labels"],
            "tid": pid2info[node.event["pid"]]["threads"][node.event["tid"]],
            "args": {
                "name": f'{node.event["name"]}_{self.trace_name_cnt[node.event["name"]]}'
            }
        }


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
    [(e_id, e) for e_id, e in enumerate(traces["traceEvents"]) if (isinstance(e["pid"], int) or e["pid"].isdigit()) and e["ph"] == "X"],
    key=cmp_to_key(event_cmp_func))

trace_tree_dict = {}
for e_id, event in sorted_traces:
    if event["pid"] not in trace_tree_dict:
        trace_tree_dict[event["pid"]] = {}
    if event["tid"] not in trace_tree_dict[event["pid"]]:
        trace_tree_dict[event["pid"]][event["tid"]] = TraceTree()
    if event["ph"] == "X":
        # print(event["pid"], event["tid"], event["ts"], event["name"][:100])
        trace_tree_dict[event["pid"]][event["tid"]].add_event(e_id, event)

# Show trace trees
# for pid in trace_tree_dict.keys():
#     for tid in trace_tree_dict[pid].keys():
#         print(f"\n{pid} {tid}")
#         trace_tree_dict[pid][tid].show()
        

rst_traces = []
for pid in trace_tree_dict.keys():
    if pid2info[pid]["process_labels"] != "CPU":
        continue
    for tid in trace_tree_dict[pid].keys():
        rst_traces.extend(trace_tree_dict[pid][tid].gen_traces())
        
output_file = "tmp.json"
print(f"Dump {len(rst_traces)} events to {output_file}") 
with open(output_file, 'w') as fp:
    json.dump({
        "traceEvents": rst_traces
    }, fp, indent=4)

with open("name2decision_cfg.json", 'w') as fp:
    json.dump(name2decision, fp, indent=4)