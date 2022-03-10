from abc import abstractmethod
from turtle import forward

import torch
import torch.nn as nn
from torchvision import models
from .common import Backbone

class BaseConv(nn.Module):
    def __init__(self, in_dim):
        super(BaseConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 32, kernel_size = 3, stride = 1), # (in_dim, 64) -> (32, 62)
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size = 3, stride = 1), # (32, 62) -> (32, 60)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2), # (32, 60) -> (32, 30)
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1), # (32, 30) -> (64, 38)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1), # (64, 38) -> (64, 36)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2), # (64, 36) -> (64, 18)
        )
    def forward(self, X):
        return self.net(X)


class FontIdentification(nn.Module):
    def __init__(self, in_dim, feature_dim, n_hidden, drop = 0.2):
        super(FontIdentification, self).__init__()
        self.backbone = BaseConv(in_dim)

        self.head = nn.Sequential(
            nn.Linear(feature_dim, n_hidden),
            nn.Dropout2d(drop),
            nn.ReLU(),
            nn.Linear(n_hidden, 2)
        )
        self.__set_params()

    def __set_params(self):
        self.backbone_params = [p for n, p in self.backbone.named_parameters() if 'backbone' in n]
        self.head_params = [p for n, p in self.backbone.named_parameters() if 'backbone' not in n]

    @abstractmethod
    def forward(self, X):
        pass

def get_size_feature(in_dim, in_size):
    net = BaseConv(in_dim)
    dummy = torch.zeros(1, in_dim , in_size, in_size)
    out = net(dummy)
    return out.flatten().size(0)


class FontIdentification2Flow(FontIdentification):
    def __init__(self, in_dim, in_size, n_hidden, drop = 0.2):
        feature_dim = get_size_feature(in_dim, in_size)
        super(FontIdentification2Flow, self).__init__(in_dim, feature_dim * 2, n_hidden, drop)
    
    def forward(self, X1, X2):
        '''
            X1, X2: (batch, channel, h, w)
        '''
        batch_size = X1.shape[0]
        input_X = torch.cat((X1, X2), dim = 0) # (batch*2, channel, h, w)
        output_X = self.backbone(input_X)
        output_X = output_X.reshape(batch_size * 2, -1)

        input_head = torch.cat((output_X[:batch_size], output_X[batch_size:]), dim = -1)
        output = self.head(input_head)
        return output

class FontIdentification1Flow(FontIdentification):
    def __init__(self, in_dim, in_size, n_hidden, drop = 0.2):
        feature_dim = get_size_feature(in_dim, in_size)
        super(FontIdentification1Flow, self).__init__(in_dim*2, feature_dim, n_hidden, drop)
    
    def forward(self, X1, X2):
        '''
            X1, X2: (batch, channel, h, w)
        '''
        batch_size = X1.shape[0]
        input_X = torch.cat((X1, X2), dim = 1) # (batch, channel*2, h, w)
        output_X = self.backbone(input_X)
        output = self.head(output_X.view(batch_size, -1))
        return output

    