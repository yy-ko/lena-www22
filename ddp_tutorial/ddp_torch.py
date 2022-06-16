from __future__ import print_function

import argparse
import random, os
import numpy as np
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torchvision.models as models

from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import StepLR




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# for Reproducibility
def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)




def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1, keepdim=True)  
            correct += predictions.eq(labels.view_as(predictions)).sum().item()

    test_accuracy = correct / len(test_loader.dataset)

    return test_accuracy



def main():
    parser = argparse.ArgumentParser(description='PyTorch DDP Example')

    parser.add_argument("--num_workers", type=int, default=1, help="The total number of workers.")
    parser.add_argument("--backend", type=str, default='nccl', help="Backend for Distributed PyTorch: nccl, gloo, mpi")
    parser.add_argument('--model', type=str, default='RESNET18', help='Name of Model.')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Name of dataset')

    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size for one process.")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate.")

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='How many batches to wait before logging training status')

    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--model_dir', type=str, default='../trained_models', help='Path for saving the trained model')
    args = parser.parse_args()

    set_random_seeds(args.seed)
    local_rank = args.local_rank

    # Initializes the distributed backend, taking care of sychronizing workers
    dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size()

    if local_rank == 0:
        logging.info('====================DISTRIBUTED DATA PARALLELISM (DDP)======================')
        logging.info('==============================TRAIN INFO START==============================')
        logging.info('  - TOTAL WORLD SIZE: %d' % (world_size))
        logging.info('  - BACKEND = %s' % (args.backend))
        logging.info('  - NUM WORKERS (GPUS) PER NODE = %s' % (args.num_workers))
        logging.info('  ' )
        logging.info('  - TRAINING MODEL = %s' % (args.model))
        logging.info('  - DATASET = %s' % (args.dataset))
        logging.info('  - BATCH SIZE = ' + str(args.batch_size))
        logging.info('  - NUM EPOCHS = ' + str(args.num_epochs))
        logging.info('  - LEARNING RATE = ' + str(args.learning_rate))
        logging.info('  ' )
        #  logging.info('  - DATA_DIR = ' + args.data_dir)
        #  logging.info('  - OUTPUT_DIR = ' + args.output_dir)
        logging.info('  - LOGGING INTERVAL = %d' % (args.log_interval))
        #  logging.info('  - WEIGHT_DECAY = ' + str(args.weight_decay))
        logging.info('============================== TRAIN INFO END ==============================')




    # Encapsulate the model on the GPU assigned to the current process
    device = torch.device("cuda:{}".format(local_rank))


    # Define the training model and Prepare dataset and dataloader
    if args.dataset == 'MNIST':
        model = Net()

        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

        train_set = datasets.MNIST('../data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST('../data', train=False, download=True, transform=transform)

    elif args.dataset == 'CIFAR10':
        model = models.resnet18(pretrained=False)
        weight_decay = 5e-4

        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train_set = datasets.CIFAR10("../../data", train=True, download=False, transform=transform) 
        test_set = datasets.CIFAR10("../../data", train=False, download=False, transform=transform)

    elif args.dataset == 'IMAGENET':
        model = models.resnet50(pretrained=False)
        weight_decay = 1e-4

        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train_set = datasets.CIFAR10("../data", train=True, download=False, transform=transform) 
        test_set = datasets.CIFAR10("../data", train=False, download=False, transform=transform)



    #  if resume == True:
        #  map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        #  ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location))


    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset=train_set)




    model = model.to(device)

    # model = Net()
    #  model = models.resnet50(pretrained=False)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    train_loader = DataLoader(dataset=train_set,
                        batch_size=args.batch_size,
                        sampler=train_sampler,
                        num_workers=4)
    test_loader = DataLoader(dataset=test_set,
                        batch_size=128,
                        shuffle=False,
                        num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=weight_decay)

    #  scheduler = StepLR(optimizer, step_size=5, gamma=0.1)


    # Training Loop
    start_time = time.time()
    total_start_time = start_time
    max_accuracy = 0

    for epoch in range(args.num_epochs):
        ddp_model.train()
        train_sampler.set_epoch(epoch)

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            if local_rank == 0 and batch_idx % args.log_interval == 0:
                elapsed_time = time.time() - start_time
                img_per_sec = (args.log_interval * args.batch_size) / elapsed_time

                logging.info('Train Epoch: {} [{}/{} ({:.6f}%)]\tLoss: {:.4f}\tImage/sec: {:.2f}'.format(
                    epoch, 
                    #  batch_idx * len(inputs), 
                    batch_idx * args.batch_size * world_size, 
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), 
                    loss.item(),
                    img_per_sec))

                start_time = time.time()

        #  if local_rank == 0 and epoch % args.test_epochs:
        if local_rank == 0:
            test_accuracy = evaluate(model=ddp_model, device=device, test_loader=test_loader)

            logging.info('Current Epoch: {}, Test Accuracy: {:.4f}'.format(epoch, test_accuracy))

            if test_accuracy > max_accuracy:
                max_accuracy = test_accuracy

            if args.save_model:
                torch.save(ddp_model.state_dict(), args.model_dir)

        #  scheduler.step()

    if local_rank == 0:
        total_train_time = time.time() - total_start_time
        logging.info('=============================== Training End ===============================')
        logging.info('Final Test Accuracy: {:.4f}, Total training time: {:.2f} (sec.)'.format(max_accuracy, total_train_time))
        logging.info('============================================================================')



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()







