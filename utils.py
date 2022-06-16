from __future__ import print_function

import numpy as np

import torch
import torch.optim as optim
import torch.distributed as dist

import random

# get parameter groups for optimizer
# for layer-wise lr scalining, each param corresponds to each param group

EPSILON = 1e-6

def get_train_steps(dataset, batch_size, total_epochs, warmup_ratio):
    """ Returns a learning rate function based on the user-defined lr scaling method, data and model.
    Args:

    Raises:
        Unsupported dataset and lr scaling methods.
    """
    theta = warmup_ratio / 100 # percentage
    warmup_epochs = int(total_epochs * theta)
    decaying_epochs = [30, 60, 80] # for imagenet

    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        steps_per_epoch = int(45000/batch_size)

        total_steps = total_epochs * steps_per_epoch # number of training samples except for validation samples
        warmup_steps = warmup_epochs * steps_per_epoch
        decaying_steps = []

    elif dataset == 'IMAGENET':
        steps_per_epoch = int(1281167/batch_size)

        total_steps = total_epochs * steps_per_epoch # number of training samples
        warmup_steps = warmup_epochs * steps_per_epoch

        decaying_steps = [steps_per_epoch * d_epoch for d_epoch in decaying_epochs]
    else:
        pass

    return total_steps, warmup_steps, decaying_steps




def initialize_meta_info(method, num_params):
    """ Returns a dict of meta information based on the user-defined lr scaling method, data and model.

    Args:
        method: the name of learning raet scaling method
        num_params: the number of model.parameters() (used only for LENA)

    Raises:
        Unsupported dataset and lr scaling methods.
    """
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
    """ Returns optimizer of PyTorch for large batch training.

    Args:
        model: nn.Module of PyTorch
        args: user-defined arguments (dict)

    Raises:
        Unsupported dataset and lr scaling methods.
    """

    if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
        weight_decay = 5e-4

    elif args.dataset == 'IMAGENET':
        weight_decay = 1e-4

    else:
        raise ValueError('Wrong dataset name!')



    if args.method == 'LSW' or args.method == 'AS':
        param_groups = model.parameters()

    elif args.method == 'LENA':
        param_groups = [ {'params':p, 'name':n, 'lr': args.learning_rate} for n, p in model.named_parameters()]
        #  for n, p in model.named_parameters():
            #  print ('{}: {} ({})'.format(n, type(p), p.requires_grad))

    else:
        raise ValueError('Wrong learning rate scaling name!')


    return optim.SGD(param_groups,
                        lr=args.learning_rate,
                        momentum=0.9,
                        weight_decay=weight_decay)






def local_aggregate(method, model, local_grads_list, current_phase, meta_info):
    """ Returns a locally aggregated gradients list based on the selected method.

    Args:

    Raises:
        Unsupported dataset and lr scaling methods.
    """

    if method == 'LSW': # Linear scaling with warm-up

        #  def local_aggregate(model, local_grads_list, current_phase):
            # first iteration in local aggregation
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
    """ Returns a learning rate function based on the user-defined lr scaling method, data and model.

    Args:

    Raises:
        Unsupported dataset and lr scaling methods.
    """

    for i, param in enumerate(model.parameters()):
        dist.all_reduce(local_grads_list[i], op=dist.ReduceOp.SUM)

        local_grads_list[i] /= float(num_batches_per_update)
        param.grad.data = local_grads_list[i]





def get_scaling_factor(method, num_batches_per_update, averaged_grads_list, meta_info):
    """ Returns a learning rate function based on the user-defined lr scaling method, data and model.

    Args:

    Raises:
        Unsupported dataset and lr scaling methods.
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

    # no warmup
    if warmup == 0:
        for i, param in enumerate(optimizer.param_groups):
            scale = scaling_factor[i] if scaling_factor[i] <= num_batches_per_update else num_batches_per_update # maximum degree of lr scaling
            #  optimizer.param_groups[i]['lr'] = scale * base_lr * (num_trained_batches / warmup_steps)
            #  optimizer.param_groups[i]['lr'] = scale * base_lr * ( 0.01 ** (num_trained_batches / total_steps) )
            optimizer.param_groups[i]['lr'] = scale * base_lr * decay

    # fixed warmup
    elif warmup == 1:
        if num_trained_batches < warmup_steps:
            for i, param in enumerate(optimizer.param_groups):
                scale = scaling_factor[i] if scaling_factor[i] <= num_batches_per_update else num_batches_per_update # maximum degree of lr scaling
                #  optimizer.param_groups[i]['lr'] = scale * base_lr * (num_trained_batches / warmup_steps) * ( 0.01 ** (num_trained_batches / total_steps) )
                optimizer.param_groups[i]['lr'] = scale * base_lr * (num_trained_batches / warmup_steps) * decay 

        else:
            for i, param in enumerate(optimizer.param_groups):
                if random.randrange(10) < 3:
                    scale = scaling_factor[i] if scaling_factor[i] <= num_batches_per_update else num_batches_per_update # maximum degree of lr scaling
                else:
                    scale = scaling_factor[i] if meta_info['grad_var'] <= num_batches_per_update else num_batches_per_update # maximum degree of lr scaling

                #  scale = scaling_factor[i] if scaling_factor[i] <= num_batches_per_update else num_batches_per_update # maximum degree of lr scaling
                #  optimizer.param_groups[i]['lr'] = scale * base_lr * ( 0.01 ** ((num_trained_batches - warmup_steps) / (total_steps - warmup_steps)) )
                optimizer.param_groups[i]['lr'] = scale * base_lr * ( 0.01 ** (num_trained_batches / total_steps) )
                optimizer.param_groups[i]['lr'] = scale * base_lr * decay 

    # proposed warmup
    elif warmup == 2:
        for i, param in enumerate(optimizer.param_groups):
            scale = scaling_factor[i] if scaling_factor[i] <= num_batches_per_update else num_batches_per_update # maximum degree of lr scaling

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



    #  elif dataset == 'CIFAR1000':
        #  if epoch < 150:
            #  base_lr *= 1

        #  elif epoch < 225:
            #  base_lr *= 0.1

        #  else:
            #  base_lr *= 0.01

        #  if warmup == 0:
            #  for i, param in enumerate(optimizer.param_groups):
                #  scale = scaling_factor[i] if scaling_factor[i] <= num_batches_per_update else num_batches_per_update # maximum degree of lr scaling
                #  optimizer.param_groups[i]['lr'] = scale * base_lr


        #  elif warmup == 1:
            #  if num_trained_batches < warmup_steps:
                #  for i, param in enumerate(optimizer.param_groups):
                    #  scale = scaling_factor[i] if scaling_factor[i] <= num_batches_per_update else num_batches_per_update # maximum degree of lr scaling
                    #  optimizer.param_groups[i]['lr'] = scale * base_lr * (num_trained_batches / warmup_steps)
            #  else:
                #  for i, param in enumerate(optimizer.param_groups):
                    #  scale = scaling_factor[i] if scaling_factor[i] <= num_batches_per_update else num_batches_per_update # maximum degree of lr scaling
                    #  optimizer.param_groups[i]['lr'] = scale * base_lr

            #  #  elif epoch < 225:
                #  #  for i, param in enumerate(optimizer.param_groups):
                    #  #  scale = scaling_factor[i] if scaling_factor[i] <= num_batches_per_update else num_batches_per_update # maximum degree of lr scaling
                    #  #  optimizer.param_groups[i]['lr'] = scale * base_lr * 0.1

            #  #  else:
                #  #  for i, param in enumerate(optimizer.param_groups):
                    #  #  scale = scaling_factor[i] if scaling_factor[i] <= num_batches_per_update else num_batches_per_update # maximum degree of lr scaling
                    #  #  optimizer.param_groups[i]['lr'] = scale * base_lr * 0.01


        #  elif warmup == 2:

            #  for i, param in enumerate(optimizer.param_groups):
                #  scale = scaling_factor[i] if scaling_factor[i] <= num_batches_per_update else num_batches_per_update # maximum degree of lr scaling

                #  if meta_info['warmup_progress'][i] < warmup_steps:
                    #  optimizer.param_groups[i]['lr'] = scale * base_lr * (meta_info['warmup_progress'][i] / warmup_steps)
                #  else:
                    #  optimizer.param_groups[i]['lr'] = scale * base_lr


                #  if scale == num_batches_per_update: # LSW
                    #  meta_info['warmup_progress'][i] += meta_info['grad_var']
                #  else:
                    #  meta_info['warmup_progress'][i] += scale









# to be deprecated
#  def get_distributed_optimizer_fn(method, dataset, world_size, num_iterations, total_epochs, batch_size, learning_rate):
    #  """ Returns a learning rate function based on the user-defined lr scaling method, data and model.

    #  Args:

    #  Raises:
        #  Unsupported dataset and lr scaling methods.
    #  """

    #  # in our setting, global batch size is multiplied by (# of workers and # of local iterations)
    #  num_batches_per_update= world_size * num_iterations # number of per-worker batches for an update
    #  base_lr = learning_rate

    #  warmup_epochs = int(total_epochs * 0.055)

    #  if method == 'LSW': # Linear scaling with warm-up
        #  if dataset == 'CIFAR10':
            #  steps_per_epoch = int(45000/batch_size)

            #  total_steps = total_epochs * steps_per_epoch # number of training samples except for validation samples
            #  warmup_steps = warmup_epochs * steps_per_epoch 

            #  # 0. just scale the learning rate by multiplying world_size
            #  def get_distributed_optimizer(model, local_grads_list, optimizer, num_updates, meta_info):
                #  num_trained_batches = num_updates * num_batches_per_update

                #  # average locally aggregated gradients and allreduce them
                #  #  for i, param in enumerate(model.parameters()):
                    #  #  grad = local_grads_list[i] / float(num_iterations)

                    #  #  dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                    #  #  grad /= float(world_size)
                    #  #  param.grad.data = grad

                #  if num_trained_batches < warmup_steps:
                    #  optimizer.param_groups[0]['lr'] = num_batches_per_update * base_lr * (num_trained_batches / warmup_steps)
                #  else:
                    #  optimizer.param_groups[0]['lr'] = num_batches_per_update * base_lr * ( 0.01 ** ((num_trained_batches - warmup_steps) / (total_steps - warmup_steps)) )
                #  return optimizer

        #  elif dataset == 'IMAGENET':
            #  pass
        #  else:
            #  pass

        #  return get_distributed_optimizer


    #  elif method == 'AS': # AdaScale with gradual warm-up
        #  # 1. compute square sum of gradients
        #  # 2. (all)reduce the square sum of gradients --> get the mean of square sum of gradients (upper side)
        #  # 3. allreduce the gradients --> get the mean of gradients
        #  # 4. compute the gain ratio
        #  # 5. adaptively scaling the learning rate by the gain ratio

        #  if dataset == 'CIFAR10':
            #  def get_distributed_optimizer(model, local_grads_list, optimizer, num_updates, meta_info):

                #  # 1. compute square sum of gradients
                #  square_sum_grads = 0
                #  for param in model.parameters():
                    #  grad = param.grad.data.clone()
                    #  square_sum_grads += torch.sum(grad * grad)
                    #  #  torch.sum(torch.square(grad))

                #  dist.all_reduce(square_sum_grads, op=dist.ReduceOp.SUM)
                #  average_square_sum = square_sum_grads / float(world_size) # 1/S * sum of (g^i**2)

                #  for param in model.parameters():
                    #  dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    #  param.grad.data /= float(world_size)

                #  square_sum_average = 0
                #  for param in model.parameters():
                    #  grad = param.grad.data.clone()
                    #  square_sum_average += torch.sum(grad * grad)

                #  gain_ratio = average_square_sum / square_sum_average


                #  if train_iter < WARMUP_CIFAR:
                    #  optimizer.param_groups[0]['lr'] = gain_ratio * base_lr * (train_iter/WARMUP_CIFAR)
                #  else:
                    #  optimizer.param_groups[0]['lr'] = gain_ratio * base_lr * ( 0.0133 ** ((train_iter-WARMUP_CIFAR)/(TOTAL_CIFAR-WARMUP_CIFAR)) )

                #  return optimizer

        #  elif dataset == 'IMAGENET':
            #  pass
        #  else:
            #  pass

        #  return get_distributed_optimizer


    #  elif method == 'LENA':
        #  pass
    #  else:
        #  raise ValueError('Wrong learning rate scaling name!')




#  def _allreduce_gradients(model, world_size):
    #  for param in model.parameters():
        #  dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        #  param.grad.data /= float(world_size)









#  # add user-defined new lr scaling rulesa...






#  #  def get_lr_scheduler(method='exp', ):
    #  #  if method == 'exp':
        #  #  pass
    #  #  elif method == 'step':
        #  #  scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    #  #  else:
        #  #  raise ValueError('Wrong learning rate scheduler name!')

    #  #  return scheduler
