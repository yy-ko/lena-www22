# Not All Layers Are Equal: A Layer-Wise Adaptive Approach toward Large-Scale DNN Training
This repository provides an implementation of *LENA* as described in the following paper: [Paper Link](https://yy-ko.github.io/assets/files/WWW22-lena-paper.pdf)
> "Not All Layers Are Equal: A Layer-Wise Adaptive Approach toward Large-Scale DNN Training"<br>
> Yunyong Ko, Dongwon Lee, and Sang-Wook Kim, The ACM Web Conference (WWW 2022)<br>


### How to run
Run with the 'torch.distributed.launch command':
  ```
  python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="192.168.0.1" run.py \
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

````num_workers````: the number of workers (# of GPUs) <br>
````num_iterations````: the number of local iterations within a node <br>
````method````: the name of the learning rate scaling scheduler <br>
````batch_size````: batch size per worker (GPU) <br>
````num_epochs````: the total number of epochs <br>
````learning_rate````: base learning rate <br>
````warmup````: the type of a warmup method (0: no warmup, 1: fixed warmup, 2: layer-wise train-aware warmup) <br>
````warmup_ratio````: the percentage of the training for warmup period <br>



## Citation
Please cite our paper if you have used the code in your work. You can use the following BibTex citation:
```
@inproceedings{ko2022not,
  title={Not All Layers Are Equal: A Layer-Wise Adaptive Approach Toward Large-Scale DNN Training},
  author={Ko, Yunyong and Lee, Dongwon and Kim, Sang-Wook},
  booktitle={Proceedings of the ACM Web Conference (WWW) 2022},
  pages={1851--1859},
  year={2022}
}
```
