import os, sys
import json
from functools import cmp_to_key
import re
import argparse

from flow import Flow
from trace_tree import TraceTree

parser = argparse.ArgumentParser(description="Troch Trace Analysis",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--torch_trace", type=str, default=None, help="torch trace directory")
parser.add_argument("--output", type=str, default=None, help="output path")
parser.add_argument("--compare", action="store_true", help="Set true to generate a JSON file that contains both the processed traces and the original torch traces for comparison")

args = parser.parse_args()

def load_torch_profiler_rst(torch_profiler_path):
    print(f"Torch Profiler Trace loading at {torch_profiler_path[:50]} ... ")
    with open(torch_profiler_path, 'r') as fp:
        traces = json.load(fp)
    return traces

def load(log_dir):
    ''' Load torch profile results and runtime information we collected
    
    We record the full name and module name of operations and the execution order of those operations in the `fullname_order` field in the metadata.json. Here, full name describes the hierarchical module information of a operation, an example of a pair of full_name and module name is as follows
        "layer2/3/conv2", "<class 'torch.nn.modules.conv.Conv2d'>"
    '''
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
    torch_traces = load_torch_profiler_rst(torch_profiler_path)["traceEvents"]
    return metadata, torch_traces

metadata, torch_traces = load(args.torch_trace)
 
def extract_pid_and_flow_info(traces):
    pid2info = {}
    flow_keyed_by_id = {}
    print("Stat pid info ... ")
    for e_id, event in enumerate(traces):
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
            source_event = traces[e_id-1]
            assert source_event["pid"] == event["pid"]
            assert source_event["tid"] == event["tid"]
            assert source_event["ts"] == event["ts"]
            # TODO what's the usage of the `cat` and `name` fields for this flow evnet
            if event["id"] not in flow_keyed_by_id:
                flow_keyed_by_id[event["id"]] = Flow(traces, event["id"], event["cat"])
            flow_keyed_by_id[event["id"]].register_in_event(e_id-1)
        elif event["ph"] == "f":
            target_event = traces[e_id-1]
            assert target_event["pid"] == event["pid"]
            assert target_event["tid"] == event["tid"]
            assert target_event["ts"] == event["ts"]
            if event["id"] not in flow_keyed_by_id:
                flow_keyed_by_id[event["id"]] = Flow(traces, event["id"], event["cat"])
            flow_keyed_by_id[event["id"]].register_out_event(e_id-1)
        
    return pid2info, flow_keyed_by_id
    
pid2info, flow_keyed_by_id = extract_pid_and_flow_info(torch_traces)

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
    [(e_id, e) for e_id, e in enumerate(torch_traces) if 
        (isinstance(e["pid"], int) or e["pid"].isdigit()) and e["ph"] == "X"],
    key=cmp_to_key(event_cmp_func))
min_ts = sorted_traces[0][1]["ts"]

pid_to_trace_tree = {}
for sorted_e_id, (e_id, event) in enumerate(sorted_traces):
    if event["pid"] not in pid_to_trace_tree:
        pid_to_trace_tree[event["pid"]] = {}
    if event["tid"] not in pid_to_trace_tree[event["pid"]]:
        pid_to_trace_tree[event["pid"]][event["tid"]] = TraceTree(torch_traces, metadata, pid2info, flow_keyed_by_id)
    if event["ph"] == "X" and "ProfilerStep" not in event["name"] \
        and "#SGD.zero_grad" not in event["name"] \
        and "#SGD.step" not in event["name"]:
        # TODO(huhanpeng): why #SGD.zero_grad weird
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
        
output_file = args.output or "tmp.json"
if args.compare:
    print(f"Merge {len(rst_traces)} processed events with the original torch traces, dumped to {output_file}") 
    rst_traces.extend(torch_traces)
else:
    print(f"Dump {len(rst_traces)} events to {output_file}") 
with open(output_file, 'w') as fp:
    json.dump({
        "traceEvents": rst_traces
    }, fp, indent=4)
