#!/bin/bash
export OMP_NUM_THREADS=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=GRAPH
# export NCCL_DEBUG_SUBSYS=COLL
export NCCL_DEBUG_SUBSYS=INIT

TEST_DIR=$(dirname $(realpath $0))

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=23456 $TEST_DIR/test_dist.py

# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=3 --node_rank=1 --master_addr="192.168.1.201" --master_port=23456 env_init.py

# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=3 --node_rank=2 --master_addr="192.168.1.201" --master_port=23456 env_init.py