import json
import os
import sys
import re
from functools import cmp_to_key

def read_file(f, pid_prefix):
    with open(f, 'r') as fp:
        traces = json.load(fp)

    event_dict = {}
    for event in sorted(traces["traceEvents"], key=lambda x: x["ts"]):
        if "cat" not in event:
            # {'name': 'process_name', 'ph': 'M', 'ts': 1694424619998603, 'pid': 1127, 'tid': 0, 'args': {'name': 'python3'}}
            continue
            
        if event["cat"] == "kernel":
            event["pid"] = pid_prefix + "/" + str(event["pid"])
            if event["pid"] not in event_dict:
                event_dict[event["pid"]] = {}
            if event["tid"] not in event_dict[event["pid"]]:
                 event_dict[event["pid"]][event["tid"]] = []
                
            event_dict[event["pid"]][event["tid"]].append(event)

    return event_dict

class WorkerInfo:
    def __init__(self, rank, dp_rank, pp_rank, tp_rank, event_dict):
        self.rank = int(rank)
        self.dp_rank = int(dp_rank)
        self.pp_rank = int(pp_rank)
        self.tp_rank = int(tp_rank)
        assert len(event_dict) == 1, "Each worker (GPU) only has one CUDA pid"
        
        self.nccl_send_tid = self.nccl_recv_tid = None
        self.tid2events = next(iter(event_dict.values()))
        self.timestamp_bias = 0
    
    def decide_send_recv_tid(self):
        # Figure out which tid denote NCCL SEND
        nccl_send_recv_tids = []
        for tid, events in self.tid2events.items():
            if events[0]["name"].startswith("ncclKernel_SendRecv_"):
                nccl_send_recv_tids.append((tid, events[0]["ts"]))
        assert len(nccl_send_recv_tids) == 2, "One Send Stream and One Recv Stream"
        if self.pp_rank == 0:
            # For the first pipeline stage, send occurs before recv
            send_idx = 0 if nccl_send_recv_tids[0][1] < nccl_send_recv_tids[1][1] else 1
        else:
            # For the remaining pipeline stage, recv occurs before send
            send_idx = 1 if nccl_send_recv_tids[0][1] < nccl_send_recv_tids[1][1] else 0
        recv_idx = 1 - send_idx
        self.nccl_send_tid = nccl_send_recv_tids[send_idx][0]
        self.nccl_recv_tid = nccl_send_recv_tids[recv_idx][0]
                   
    def correct_traces(self, output_events):
        for events in self.tid2events.values():
            for event in events:
                event["ts"] += self.timestamp_bias
                output_events.append(event)
        self.timestamp_bias = 0

    @property
    def first_recv_event(self):
        assert self.nccl_recv_tid is not None
        return self.tid2events[self.nccl_recv_tid][0]
    
    @property
    def first_send_event(self):
        assert self.nccl_send_tid is not None
        return self.tid2events[self.nccl_send_tid][0]

def f_comp(pid1, pid2):
    match1 = re.search(r'R(?P<R_rank>\d+)-DP(?P<DP_rank>\d+)-PP(?P<PP_rank>\d+)-TP(?P<TP_rank>\d+)', pid1).groupdict()
    match2 = re.search(r'R(?P<R_rank>\d+)-DP(?P<DP_rank>\d+)-PP(?P<PP_rank>\d+)-TP(?P<TP_rank>\d+)', pid2).groupdict()
    return int(match1["R_rank"]) - int(match2["R_rank"])
    
print(sys.argv[1])
root, dirs, files = list(os.walk(sys.argv[1]))[0]
dirs = sorted(dirs, key=cmp_to_key(f_comp))
worker_info_dict = {}

pp_rank2rank = {}
for _dir in dirs:
    _root = os.path.join(root, _dir)
    _file = os.listdir(_root)[0]
    match = re.search(r'R(?P<R_rank>\d+)-DP(?P<DP_rank>\d+)-PP(?P<PP_rank>\d+)-TP(?P<TP_rank>\d+)', _dir).groupdict()
    if int(match["TP_rank"]) != 0:
        continue
    print(_dir)
    event_dict = read_file(os.path.join(_root, _file), _dir)
    worker_info = WorkerInfo(
        match["R_rank"],
        match["DP_rank"],
        match["PP_rank"],
        match["TP_rank"],
        event_dict
    )
    worker_info_dict[worker_info.rank] = worker_info
    pp_rank2rank[worker_info.pp_rank] = worker_info.rank

events = []  
# Traverse the pipeline and correct the traces
for pp_rank in sorted(pp_rank2rank.keys()):
    rank = pp_rank2rank[pp_rank]
    worker_info = worker_info_dict[rank]
    worker_info.decide_send_recv_tid()
        
    if worker_info.pp_rank > 0:
        first_recv_event = worker_info.first_recv_event
        prev_send_event = worker_info_dict[pp_rank2rank[worker_info.pp_rank-1]].first_send_event
        worker_info.timestamp_bias = (prev_send_event["ts"] + prev_send_event["dur"]) - \
            (first_recv_event["ts"] + first_recv_event["dur"])
    worker_info.correct_traces(output_events=events)

# events = sorted(events, key=lambda x: x["pid"])
with open(os.path.join(sys.argv[1], 'combined.json'), 'w') as fp:
    json.dump({
        "traceEvents": events,
    }, fp, indent=4)