import numpy as np

import torch
import torch.optim as optim
import torch.distributed as dist

import random

EPSILON = 1e-6

def get_scaling_factor(method, num_batches_per_update, averaged_grads_list, meta_info):
    """ Returns a learning rate function based on the user-defined lr scaling method, data and model.

    """
    if method == 'LSW':
        meta_info['scaling_factor'][0] = num_batches_per_update

        # 1) compute upside of the gain_ratio equation: 1/S * sum of (g^i**2)
        dist.all_reduce(meta_info['square_sum_grads'], op=dist.ReduceOp.SUM)
        mean_square_sum_grads = meta_info['square_sum_grads'] / float(num_batches_per_update)

        # tracking the moving average of mean_square_sum_grads
        meta_info['moving_average_top'] = (num_batches_per_update/1000) * mean_square_sum_grads + meta_info['moving_average_top'] * (1-num_batches_per_update/1000)


        # 2) compute downside of the gain_ratio equation: |E(X)|**2
        square_sum_mean_grads = 0
        for avg_grad in averaged_grads_list:
            square_sum_mean_grads += torch.sum(avg_grad * avg_grad)

        # tracking the moving average of quare_sum_mean_grads
        meta_info['moving_average_bottom'] = (num_batches_per_update/1000) * square_sum_mean_grads + meta_info['moving_average_bottom'] * (1-num_batches_per_update/1000)

        # 3) compute the value of gain ratio
        meta_info['grad_var'] = (meta_info['moving_average_top'] + EPSILON) / (meta_info['moving_average_bottom'] + EPSILON)

        return meta_info

    elif method == 'AS':
        # 1) compute upside of the gain_ratio equation: 1/S * sum of (g^i**2)
        dist.all_reduce(meta_info['square_sum_grads'], op=dist.ReduceOp.SUM)
        mean_square_sum_grads = meta_info['square_sum_grads'] / float(num_batches_per_update)

        # tracking the moving average of mean_square_sum_grads
        meta_info['moving_average_top'] = (num_batches_per_update/1000) * mean_square_sum_grads + meta_info['moving_average_top'] * (1 - num_batches_per_update/1000)


        # 2) compute downside of the gain_ratio equation: |E(X)|**2
        square_sum_mean_grads = 0
        for avg_grad in averaged_grads_list:
            square_sum_mean_grads += torch.sum(avg_grad * avg_grad)

        # tracking the moving average of quare_sum_mean_grads
        meta_info['moving_average_bottom'] = (num_batches_per_update/1000) * square_sum_mean_grads + meta_info['moving_average_bottom'] * (1 - num_batches_per_update/1000)

        # 3) compute the value of gain ratio
        meta_info['scaling_factor'][0] = (meta_info['moving_average_top'] + EPSILON) / (meta_info['moving_average_bottom'] + EPSILON)
        meta_info['grad_var'] = meta_info['scaling_factor'][0]

        return meta_info

    elif method == 'LENA':
        # for each layer,
        all_sum = 0
        for i, ssg in enumerate(meta_info['square_sum_grads']):
            # 1) compute upside of the gain_ratio equation: 1/S * sum of (g^i**2)
            dist.all_reduce(ssg, op=dist.ReduceOp.SUM)
            mean_square_sum_grad = ssg / float(num_batches_per_update)
            all_sum += mean_square_sum_grad

            # tracking the moving average of mean square sum for each gradient
            meta_info['moving_average_top'][i] = (num_batches_per_update/1000) * mean_square_sum_grad + meta_info['moving_average_top'][i] * (1-num_batches_per_update/1000)
            #  meta_info['moving_average_top'][i] = mean_square_sum_grad

        meta_info['all_top'] = (num_batches_per_update/1000) * all_sum + meta_info['all_top'] * (1-num_batches_per_update/1000)
        #  meta_info['all_top'] = all_sum 

        #  square_sum_mean_grads = 0
        all_sum = 0
        for i, avg_grad in enumerate(averaged_grads_list):
            # 2) compute downside of the gain_ratio equation: |E(X)|**2
            square_sum_mean_grad = torch.sum(avg_grad * avg_grad)
            all_sum += square_sum_mean_grad

            # tracking the moving average of quare_sum_mean_grads
            meta_info['moving_average_bottom'][i] = (num_batches_per_update/1000) * square_sum_mean_grad + meta_info['moving_average_bottom'][i] * (1-num_batches_per_update/1000)
            #  meta_info['moving_average_bottom'][i] = square_sum_mean_grad 
        meta_info['all_bottom'] = (num_batches_per_update/1000) * all_sum + meta_info['all_bottom'] * (1-num_batches_per_update/1000)
        #  meta_info['all_bottom'] = all_sum 

        # 3) compute the value of gain ratio
        for i, gr in enumerate(meta_info['scaling_factor']):
            meta_info['scaling_factor'][i] = (meta_info['moving_average_top'][i] + EPSILON) / (meta_info['moving_average_bottom'][i] + EPSILON)

        meta_info['grad_var'] = (meta_info['all_top'] + EPSILON) / (meta_info['all_bottom'] + EPSILON)

        return meta_info

    else:
        raise ValueError('Wrong learning rate scaling name!')


def set_learning_rate(dataset, optimizer, base_lr, meta_info, total_steps, warmup_steps, decaying_steps, num_updates, num_batches_per_update , epoch, warmup):

    scaling_factor = meta_info['scaling_factor']
    num_trained_batches = num_updates * num_batches_per_update

    # computing decaying factor (exponential/step-wise decaying)
    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        decay = 0.01 ** (num_trained_batches / total_steps)

    elif dataset == 'IMAGENET':
        if num_trained_batches < decaying_steps[0]:
            decay = 1
        elif num_trained_batches < decaying_steps[1]:
            decay = 0.1
        elif num_trained_batches < decaying_steps[2]:
            decay = 0.01
        else:
            decay = 0.001
    else:
        pass

    # No warmup
    if warmup == 0:
        for i, param in enumerate(optimizer.param_groups):
            scale = scaling_factor[i] if scaling_factor[i] <= num_batches_per_update else num_batches_per_update
            optimizer.param_groups[i]['lr'] = scale * base_lr * decay

    # Fixed warmup
    elif warmup == 1:
        if num_trained_batches < warmup_steps:
            for i, param in enumerate(optimizer.param_groups):
                scale = scaling_factor[i] if scaling_factor[i] <= num_batches_per_update else num_batches_per_update
                optimizer.param_groups[i]['lr'] = scale * base_lr * (num_trained_batches / warmup_steps) * decay 

        else:
            for i, param in enumerate(optimizer.param_groups):
                if random.randrange(10) < 3:
                    scale = scaling_factor[i] if scaling_factor[i] <= num_batches_per_update else num_batches_per_update
                else:
                    scale = scaling_factor[i] if meta_info['grad_var'] <= num_batches_per_update else num_batches_per_update

                optimizer.param_groups[i]['lr'] = scale * base_lr * ( 0.01 ** (num_trained_batches / total_steps) )
                optimizer.param_groups[i]['lr'] = scale * base_lr * decay 

    # Layer-wise Train-aware Warmup (Proposed)
    elif warmup == 2:
        for i, param in enumerate(optimizer.param_groups):
            scale = scaling_factor[i] if scaling_factor[i] <= num_batches_per_update else num_batches_per_update

            if meta_info['warmup_progress'][i] < warmup_steps:
                optimizer.param_groups[i]['lr'] = scale * base_lr * (meta_info['warmup_progress'][i] / warmup_steps) * decay
            else:
                optimizer.param_groups[i]['lr'] = scale * base_lr * decay

                if meta_info['warmup_endpoint'][i] == 0:
                    meta_info['warmup_endpoint'][i] = epoch


            if scale == num_batches_per_update: # LSW
                meta_info['warmup_progress'][i] += meta_info['grad_var']
            else:
                meta_info['warmup_progress'][i] += scale


    return True




