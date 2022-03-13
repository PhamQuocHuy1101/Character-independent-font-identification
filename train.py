import time
import os
import argparse

import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.model_dataset import FontDataset
from data.transform import to_tensor, augmenter

import config as cf
import utils
from network import load_model

def train(cf):
    device = torch.device(cf.device)
    model = load_model(cf.model)
    model.to(device = device)
    if cf.model['freeze']:
        param_dict = [
            {'params': model.head_params}
        ]
    else:
        param_dict = [
            {'params': model.backbone_params, 'lr': cf.optim['lr'] * 0.1},
            {'params': model.head_params}
        ]

    optimizer = optim.Adam(params = param_dict, lr = cf.optim['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = cf.optim['step'], gamma=cf.optim['gamma'])

    train_df = pd.read_csv(cf.data['train_csv'])
    val_df = pd.read_csv(cf.data['val_csv'])

    train_data = FontDataset(cf.data['dir_path'], train_df.sample_1, train_df.sample_2, train_df.font_label, augmenter)
    val_data = FontDataset(cf.data['dir_path'], val_df.sample_1, val_df.sample_2, val_df.font_label, to_tensor)

    train_loader = DataLoader(train_data, batch_size = cf.optim['batch_size'], shuffle = True)
    val_loader = DataLoader(val_data, batch_size = 8, shuffle = True)

    # train
    start = time.time()

    # class_weight = torch.tensor()
    criterion = nn.CrossEntropyLoss()
    best = -1
    if cf.optim['continue_training'] and os.path.exists(cf.model['checkpoint']):
        checkpoint = torch.load(cf.model['checkpoint'], map_location = device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best = checkpoint['val_accuracy']

    for epoch in range(cf.optim['n_epoch']):
        print("Epoch ============================ {}".format(epoch))
        model.train()
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
                    'optimizer': optimizer.state_dict(),
                    'val_accuracy': best
                }, cf.model['checkpoint'])
        
        scheduler.step()
        print('Train loss: {}, val loss {}, val accuracy {}'.format(running_loss, val_loss, val_accuracy))

    print("Store at: ", cf.model['checkpoint'])
    print("Process time: ", time.time() - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training")
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--device', type=str, required=True, help='device')
    args = parser.parse_args()
    
    cf = utils.Config(args.config)
    cf.device = args.device
    train(cf)
