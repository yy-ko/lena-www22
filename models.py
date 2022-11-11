from __future__ import print_function


import torch
import torch.nn as nn
import torch.nn.functional as F



# Add the user-defined model here


def get_model(model_name, pretrained=False):
    if model_name == 'RESNET18':
        return models.resnet18(pretrained=pretrained) 
    elif model_name == 'RESNET50':
        return models.resnet50(pretrained=pretrained) 
    else:
        raise ValueError('Wrong model name!')
    #  elif model_name == 'custom': # you can define a new model here and use it
        #  return UserDefinedModel()
