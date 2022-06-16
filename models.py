from __future__ import print_function


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



# Add the user-defined model here
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


def get_model(model_name, pretrained=False):
    if model_name == 'RESNET18':
        return models.resnet18(pretrained=pretrained) 
    elif model_name == 'RESNET50':
        return models.resnet50(pretrained=pretrained) 
    elif model_name == 'DeepSpeech':
        return models.resnet50(pretrained=pretrained) 
    elif model_name == 'custom':
        return Net()
    else:
        raise ValueError('Wrong model name!')