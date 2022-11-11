# Not All Layers Are Equal: A Layer-Wise Adaptive Approach toward Large-Scale DNN Training

This repository provides an implementation of *LENA* as described in the following paper:
> Not All Layers Are Equal: A Layer-Wise Adaptive Approach toward Large-Scale DNN Training<br>
> Yunyong Ko, Dongwon Lee, and Sang-Wook Kim<br>
> The ACM Web Conference (WWW 2022)<br>


## Usage
Run by:
> num_workers: the number of workers (# of GPUs)
> num_iterations: the number of local iterations within a node
> method: the name of the learning rate scaling scheduler
> batch_size: batch size per worker (GPU)
> num_epochs: the total number of epochs
> learning_rate: base learning rate
> warmup: the type of a warmup method (0: no warmup, 1: fixed warmup, 2: layer-wise train-aware warmup)
> warmup_ratio: the percentage of the training for warmup period

  ```
  python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="192.168.0.45" run.py \
  --num_workers=2 \
  --num_iterations=16 \
  --method=LENA \
  --batch_size=128 \
  --num_epochs=100 \
  --learning_rate=0.1 \
  --warmup=2 \
  --warmup_ratio=5 \
  --dataset=CIFAR10 \
  --model=RESNET18 \
  --data_path=/DATAPATH
  ```  
