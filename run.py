import argparse
import random, os, sys
import numpy as np
import time
import logging
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchvision.models as models

from torch.optim.lr_scheduler import StepLR

import dataloaders, utils, lr_scaling


warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


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
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            #  predictions = outputs.argmax(dim=1, keepdim=True)  
            #  correct += predictions.eq(labels.view_as(predictions)).sum().item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).double().sum().item()

    test_accuracy = correct / total

    return test_accuracy



def get_model(model_name, pretrained=False):
    if model_name == 'RESNET18':
        return models.resnet18(pretrained=pretrained) 
    elif model_name == 'RESNET50':
        return models.resnet50(pretrained=pretrained) 
    else:
        raise ValueError('Wrong model name!')
    # you can define a new model and call it here
    #  elif model_name == 'custom': 
        #  return UserDefinedModel()


def main():
    parser = argparse.ArgumentParser(description='Distributed PyTorch Example')

    parser.add_argument("--num_workers", type=int, default=1, help="The number of workers per node.")
    parser.add_argument("--num_iterations", type=int, default=1, help="The number of iterations of workers without zero_grad().")
    parser.add_argument("--method", type=str, default='LSW', help="LR scaling method for large batch training.")
    parser.add_argument("--backend", type=str, default='nccl', help="Backend for Distributed PyTorch: nccl, gloo, mpi")

    parser.add_argument('--model', type=str, default='RESNET18', help='Name of Model.')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Name of dataset.')
    parser.add_argument('--data_path', type=str, default='/data', help='Data path.')

    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size for one process.")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--warmup", type=int, default=2, help="Type of warmup: 0 (no warmup), 1 (fixed gradual warmup), 2 (training-aware warmup)")
    parser.add_argument("--warmup_ratio", type=float, default=5, help="Percentage of warmup ratio")

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)')

    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--model_dir', type=str, default='../trained_models', help='Path for saving the trained model')

    args = parser.parse_args()


    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist.init_process_group(backend=args.backend)
    world_size = dist.get_world_size()
    local_rank = args.local_rank
    global_rank = dist.get_rank()
    set_random_seeds(args.seed)


    # Summary of training information
    if global_rank == 0:
        logging.info('===============================TRAIN INFO START===============================')
        logging.info('  - TOTAL WORLD SIZE: %d' % (world_size))
        logging.info('  - BACKEND = %s' % (args.backend))
        logging.info('  ' )
        logging.info('  - LR SCALING METHOD = %s' % (args.method))
        logging.info('  ' )
        logging.info('  - TRAINING MODEL = %s' % (args.model))
        logging.info('  - DATASET = %s' % (args.dataset))
        logging.info('  - TOTAL BATCH SIZE (N * B * I) = ' + str(args.batch_size*args.num_iterations*args.num_workers))
        logging.info('      - NUM WORKERS PER NODE = %s' % (args.num_workers))
        logging.info('      - PER WORKER BATCH SIZE = ' + str(args.batch_size))
        logging.info('      - PER WORKER ITERATION = ' + str(args.num_iterations))
        logging.info('  - NUM EPOCHS = ' + str(args.num_epochs))
        logging.info('  - BASE LEARNING RATE = ' + str(args.learning_rate))
        warmup = { args.warmup == 0: 'No Warmup', args.warmup == 2: 'Train-Aware Warmup'}.get(True, 'Fixed Warmup')
        logging.info('  - LEARNING RATE WAMRUP (%d) = %s' % (args.warmup, str(warmup)))
        logging.info('  - WAMRUP PERIOD = %s' % (str(args.warmup_ratio)))
        logging.info('=============================== TRAIN INFO END ===============================')





    # Encapsulate the model on the GPU assigned to the current process
    # Get the model and data loaders
    device = torch.device("cuda:{}".format(local_rank))
    model = get_model(args.model)
    model = model.to(device)

    train_loader, test_loader = dataloaders.get_dataset(args.dataset,
                                                        args.data_path,
                                                        args.batch_size,
                                                        world_size)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = utils.get_optimizer(model, args)


    # ######################################################## #
    #                       Training Loop                      #
    # ######################################################## #

    start_time = time.time()
    total_start_time = start_time
    max_accuracy = 0

    num_batches = 0
    num_updates = 0

    local_grads_list = []

    num_batches_per_update = world_size * args.num_iterations # number of per-worker batches for an update
    total_steps, warmup_steps, decaying_steps = utils.get_train_steps(args.dataset, args.batch_size, args.num_epochs, args.warmup_ratio)

    num_params = sum(1 for _ in model.parameters()) # compute the number of parameter tensors
    meta_info = utils.initialize_meta_info(args.method, num_params)

    if global_rank == 0:
        logging.info('')
        logging.info('=============================== Training Start ===============================')
        logging.info('\tEpoch\tStep\tTrain Acc.\tTest Acc.\tLoss\tImg/sec')

    for epoch in range(args.num_epochs):

        model.train()
        train_loader.sampler.set_epoch(epoch)

        train_correct = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # forward and backward passes
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # evaluate training accuracy
            predictions = outputs.argmax(dim=1, keepdim=True)
            train_correct += predictions.eq(labels.view_as(predictions)).sum().item()


            # ################################################################ #
            #   iterating local aggregation: our trick to increase batchsize   #
            # ################################################################ #

            current_phase = num_batches % args.num_iterations
            local_grads_list, meta_info = utils.local_aggregate(args.method,
                                                                model,
                                                                local_grads_list,
                                                                current_phase, meta_info)

            if current_phase == (args.num_iterations-1):
                utils.local_average_and_allreduce(model, local_grads_list, num_batches_per_update)
                meta_info = lr_scaling.get_scaling_factor(args.method, num_batches_per_update, local_grads_list, meta_info)

                lr_scaling.set_learning_rate(args.dataset, optimizer, args.learning_rate, meta_info, total_steps, warmup_steps, decaying_steps, num_updates, num_batches_per_update, epoch+1, args.warmup)
                optimizer.step()

                num_updates += 1
                local_grads_list = []

            optimizer.zero_grad()
            num_batches += 1


        # at each epoch, evaluate the test accuracy and log the progress
        if global_rank == 0:
            elapsed_time = time.time() - start_time
            train_accuracy = train_correct / len(train_loader.dataset) * world_size

            test_accuracy = evaluate(model=model, device=device, test_loader=test_loader)
            if test_accuracy > max_accuracy:
                max_accuracy = test_accuracy

            img_per_sec = len(train_loader.dataset) / elapsed_time
            current_lr = optimizer.param_groups[0]['lr']

            logging.info('\t{}\t{}\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t{:.2f}'.format(
                (epoch+1),
                num_updates,
                train_accuracy,
                test_accuracy,
                loss.item(),
                img_per_sec)
                )
            start_time = time.time()


    if global_rank == 0:
        total_train_time = time.time() - total_start_time
        logging.info('')
        logging.info('=============================== Training End ===============================')
        logging.info('Final Test Accuracy: {:.4f}, Total training time: {:.2f} (sec.)'.format(max_accuracy, total_train_time))
        logging.info('============================================================================')



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()

