import time
import os
import argparse

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDL
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm


from data.model_dataset import FontDataset
from data.transform import to_tensor, augmenter
import config as cf
import utils
from network import load_model

'''
    Only 1 node, multi gpus
'''

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def set_process(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank = rank, world_size = world_size)

def train(rank, cf):
    # init process
    if cf.n_gpu > 1:
        set_process(rank, cf.n_gpu)

    device = torch.device(rank)
    model = load_model(cf.model)
    model.to(device = device)
     
    if cf.n_gpu > 1:
        model = DDL(model, device_ids = [rank], output_device = rank)

    optimizer = optim.Adam(params = model.parameters(), lr = cf.optim['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = cf.optim['step'], gamma=cf.optim['gamma'])

    train_df = pd.read_csv(cf.data['train_csv'])
    train_data = FontDataset(cf.data['dir_path'], train_df.sample_1, train_df.sample_2, train_df.labels, augmenter)
    train_sample = DistributedSampler(dataset = train_data) if cf.n_gpu > 1 else None
    train_loader = DataLoader(train_data, 
                                batch_size = cf.optim['batch_size'], 
                                shuffle = False, 
                                sampler = train_sample,
                                pin_memory=True,
                                drop_last=True)
    if rank == 0:
        val_df = pd.read_csv(cf.data['val_csv'])
        val_data = FontDataset(cf.data['dir_path'], val_df.sample_1, val_df.sample_2, val_df.labels, to_tensor)
        val_loader = DataLoader(val_data, batch_size = 8, shuffle = False)

    # train
    start = time.time()

    # class_weight = torch.tensor()
    criterion = nn.CrossEntropyLoss()
    best = -1

    model.train()
    for epoch in range(cf.optim['n_epoch']):
        if rank == 0:
            print("Epoch ============================ {}".format(epoch))
            train_sample.set_epoch(epoch)
        
        running_loss = 0.0
        for img_1, img_2, labels in tqdm(train_loader):
            img_1 = img_1.to(device)
            img_2 = img_2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(img_1, img_2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        running_loss /= len(train_loader)
        if rank == 0:
            model.eval()
            with torch.no_grad():
                val_loss, val_accuracy = 0.0, 0.0
                for img_1, img_2, labels in val_loader:
                    img_1 = img_1.to(device = device)
                    img_2 = img_2.to(device = device)
                    labels = labels.to(device = device)

                    outputs = model(img_1, img_2)
                    val_loss += criterion(outputs, labels).item()
                    y_predict = torch.softmax(outputs, dim = 1).argmax(dim = 1)
                    val_accuracy += torch.sum(y_predict == labels).item()
                    
                val_loss /= len(val_loader)
                val_accuracy /= len(val_data)
                if best == -1 or val_loss <= best:
                    print('Store')
                    best = val_loss
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': model.state_dict(),
                        'val_accuracy': best
                    }, cf.model['checkpoint'])
            model.train()
            print('Train loss: {}, val loss {}, val accuracy {}'.format(running_loss, val_loss, val_accuracy))
        
        scheduler.step()

    print("Store at: ", cf.model['checkpoint'])
    print("Process time: ", time.time() - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training")
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--device', type=str, required=True, help='device')
    args = parser.parse_args()
    
    cf = utils.Config(args.config)

    if torch.cuda.is_available():
        set_random_seeds(cf.get('seed', 0))

    if args.device == 'cpu':
        cf.device = 'cpu'
        train(cf)
    else:
        cf.n_gpu = len(args.device.split(','))
        cf.optim['batch_size'] = int(cf.optim['batch_size'] / cf.n_gpu)
        mp.spawn(train, nprocs=cf.n_gpu, args=(cf, ))
