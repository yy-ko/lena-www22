import numpy as np

import torch
import torch.optim as optim
import torch.distributed as dist


def get_train_steps(dataset, batch_size, total_epochs, warmup_ratio):
    theta = warmup_ratio / 100 
    warmup_epochs = int(total_epochs * theta)

    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        steps_per_epoch = int(45000/batch_size)

        total_steps = total_epochs * steps_per_epoch # number of training samples except for validation samples
        warmup_steps = warmup_epochs * steps_per_epoch
        decaying_steps = []

    # elif:
        # you can define training and warmup steps for new datasets here

    else:
        raise ValueError('Wrong dataset name!')


    return total_steps, warmup_steps, decaying_steps



def initialize_meta_info(method, num_params):
    if method == 'LSW':
        meta_info = {'grad_var': 0.0,
                    'scaling_factor': [0.0],
                    'warmup_progress': [0.0],
                    'warmup_endpoint': [0],
                    'square_sum_grads': 0.0, 'moving_average_top': 0.0, 'moving_average_bottom': 0.0}
        return meta_info

    elif method == 'AS':
        meta_info = {'grad_var': 0.0,
                    'scaling_factor': [0.0],
                    'warmup_progress': [0.0],
                    'warmup_endpoint': [0],
                    'square_sum_grads': 0.0, 'moving_average_top': 0.0, 'moving_average_bottom': 0.0}
        return meta_info

    elif method == 'LENA':
        meta_info = {'grad_var': 0.0,
                    'scaling_factor': [ 0.0 for _ in range(num_params)],
                    'warmup_progress': [ 0.0 for _ in range(num_params)],
                    'warmup_endpoint': [ 0 for _ in range(num_params)],
                    'all_top': 0.0,
                    'all_bottom': 0.0,
                    'square_sum_grads': [ 0.0 for _ in range(num_params)],
                    'moving_average_top': [ 0.0 for _ in range(num_params)],
                    'moving_average_bottom': [ 0.0 for _ in range(num_params)]}
        return meta_info

    else:
        raise ValueError('Wrong learning rate scaling name!')




def get_optimizer(model, args):

    if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
        weight_decay = 5e-4
    else:
        raise ValueError('Wrong dataset name!')


    if args.method == 'LSW' or args.method == 'AS':
        param_groups = model.parameters()

    elif args.method == 'LENA':
        param_groups = [ {'params':p, 'name':n, 'lr': args.learning_rate} for n, p in model.named_parameters()]

    else:
        raise ValueError('Wrong learning rate scaling name!')

    return optim.SGD(param_groups,
                        lr=args.learning_rate,
                        momentum=0.9,
                        weight_decay=weight_decay)





def local_aggregate(method, model, local_grads_list, current_phase, meta_info):

    if method == 'LSW': # Linear scaling with warm-up

        if current_phase == 0:
            tmp_square_sum = 0
            for param in model.parameters():
                grad = param.grad.data.clone()
                local_grads_list.append(grad)

                tmp_square_sum += torch.sum(grad * grad)

            meta_info['square_sum_grads'] = tmp_square_sum

        else: # otherwise, just aggregate
            tmp_square_sum = 0
            for i, param in enumerate(model.parameters()):
                grad = param.grad.data.clone()
                local_grads_list[i] += grad
                tmp_square_sum += torch.sum(grad * grad)

            meta_info['square_sum_grads'] += tmp_square_sum

        return local_grads_list, meta_info


    elif method == 'AS': # AdaScale
        if current_phase == 0:
            tmp_square_sum = 0
            for param in model.parameters():
                grad = param.grad.data.clone()
                local_grads_list.append(grad)

                tmp_square_sum += torch.sum(grad * grad)

            meta_info['square_sum_grads'] = tmp_square_sum


        else: # otherwise, just aggregate
            tmp_square_sum = 0
            for i, param in enumerate(model.parameters()):
                grad = param.grad.data.clone()
                local_grads_list[i] += grad
                tmp_square_sum += torch.sum(grad * grad)

            meta_info['square_sum_grads'] += tmp_square_sum

        return local_grads_list, meta_info


    elif method == 'LENA':
        if current_phase == 0:
            for i, param in enumerate(model.parameters()):
                grad = param.grad.data.clone()
                local_grads_list.append(grad)

                meta_info['square_sum_grads'][i] = torch.sum(grad * grad)


        else: # otherwise, just aggregate (Layer-wise)
            tmp_square_sum = 0
            for i, param in enumerate(model.parameters()):
                grad = param.grad.data.clone()
                local_grads_list[i] += grad

                meta_info['square_sum_grads'][i] += torch.sum(grad * grad)

        return local_grads_list, meta_info

    else:
        raise ValueError('Wrong learning rate scaling name!')




def local_average_and_allreduce(model, local_grads_list, num_batches_per_update):

    for i, param in enumerate(model.parameters()):
        dist.all_reduce(local_grads_list[i], op=dist.ReduceOp.SUM)

        local_grads_list[i] /= float(num_batches_per_update)
        param.grad.data = local_grads_list[i]


