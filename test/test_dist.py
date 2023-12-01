import argparse
import os
import time
import numpy as np

import torch
import torch.distributed as dist

parser = argparse.ArgumentParser(description='PyTorch distributed training Speed test')
# NOTE: this argument definition is necessary since torch.distributed.launch will automatically pass this argument
parser.add_argument("--local-rank", type=int)

args = parser.parse_args()
args.local_rank = args.local_rank or int(os.environ['LOCAL_RANK'])

# parser.add_argument('--rank', default=0,
#                     help='rank of current process')
# parser.add_argument('--word_size', default=2,
#                     help="word size")
# parser.add_argument('--init_method', default='tcp://127.0.0.1:23456',
#                     help="init-method")
# dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', rank=args.rank, world_size=args.word_size)

dist.init_process_group(backend='nccl', init_method='env://')
backend = torch.distributed.get_backend(group=None)
rank = torch.distributed.get_rank(group=None)
world_size = torch.distributed.get_world_size(group=None)
is_nccl_available = torch.distributed.is_nccl_available()
print(f"{world_size=} {rank=} {args.local_rank=} {backend=} {is_nccl_available=}")

torch.cuda.set_device(args.local_rank)

def profile_allreduce(print=print):
    assert dist.is_initialized()
    n = 1 * 1024 * 1024 * 1024

    x = torch.randn(n).cuda(torch.cuda.current_device())
    if dist.get_rank() == 0:
        print("Profiling AllReduce... NCCL ENVs:")
        for k in os.environ:
            if k.startswith("NCCL"):
                print(f'{k} = {os.environ[k]}')
    stats = []
    for _ in range(10):
        dist.barrier()
        t = time.time()
        dist.all_reduce(x)
        torch.cuda.synchronize()
        MBps = n * 4 * 2 / (time.time() - t) / 1e9
        stats.append(MBps)
    if dist.get_rank() == 0:
        speed = np.mean(stats)
        print(f"Allreduce speed = {speed:.3f} GB/s")
        
profile_allreduce(print=print)