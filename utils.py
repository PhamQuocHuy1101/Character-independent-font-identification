import yaml
import torch

def l1(tensor):
    return tensor.flatten().abs().sum()

def l2(tensor):
    return torch.pow(tensor.flatten(), 2).sum()

def regularization(param_dict, alpha):
    l = []
    for param in param_dict:
        tensor_list = [t for t in param['params']]
        for t in tensor_list:
            l.append((alpha * t.abs() + (1.0 - alpha) * torch.pow(t, 2)).sum())
    return sum(l)

class Config():
    def __init__(self, cf):
        cf = self.__read_config(cf)
        self.__dict__.update(cf)
    
    def __read_config(self, cf_file):
        with open(cf_file, 'r') as f:
            config = yaml.safe_load(f)
            return config
    def get(self, key, d_value = None):
        return self.__dict__.get(key, d_value)
