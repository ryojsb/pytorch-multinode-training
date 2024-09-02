# PyTorch Scripts

## Multiprocess

```bash
$ python ddp-singlenode-with-mp.py --total_epochs 30 --save_every 10 --batch_size 64
```


## Torchrun

If you execute the DDP script with muliple process on a single node, then execute it.

```bash
$ torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port="<Master Port>" ddp-singlenode.py --epochs 20 --batch_size 128 --learning_rate 0.01 --pre-trained
```

In spite of executing the DDP with single GPU or multiple GPU, If you do it on a multiple node, then execute it on each nodes.

```bash


$ torchrun --nproc_per_node="<the number of GPU>" --nnodes="<the Number of Node>" --node_rank="<Global Rank>" --master_addr="<Master Address>" --master_port="<Master Port>" ddp-multinode.py
```

Example

```bash
<Node1>
$ torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="192.168.1.1" --master_port=54321 ddp-multinode.py

<Node2>
$ torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=54321 ddp-multinode.py
```


## LLM

```bash
$ torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port="<Master Port>" ddp-finetuning-torchrun.py --epochs 20 --batch_size 32
```