import argparse
import os
import torch
from PIL import Image
from tqdm.auto import tqdm

import math
import numpy as np
import pandas as pd

from network import load_model
import utils

from data.transform import to_tensor


global img_dir

class BatchIter():
    def __init__(self, df_file, batch_size):
        self.df = pd.read_csv(df_file)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)
    
    def get_batch(self, index):
        batch = []
        for i in range(index * self.batch_size, index * self.batch_size + self.batch_size):
            file = self.df.path.get(i)
            if file == None:
                break
            img = Image.open(os.path.join(img_dir, file)).convert('RGB')
            img = to_tensor(img)
            batch.append(img)
        return torch.stack(batch, dim = 0)

class Logger():
    def __init__(self, log_file, mode = 'a'):
        self.log_file = log_file
        self.log = open(log_file, mode)

    def record(self, value):
        self.log.write('\n')
        self.log.write('\t'.join([str(i) for i in value]))
    def __del__(self):
        self.log.close()
    def close(self):
        print("close logger")
        self.__del__()

def test_classify(cf, test_file, support_set, batch_size, predict_out_file):
    device = cf.device

    model = load_model(cf.model)
    checkpoint = torch.load(cf.model['checkpoint'], map_location = device)
    checkpoint_model = {k[7:]:v for k, v in checkpoint['model'].items()}
    model.load_state_dict(checkpoint_model)
    model.eval()
    print('load state dict')

    test_df = pd.read_csv(test_file)
    batch_iter = BatchIter(support_set, batch_size)

    logger = Logger(predict_out_file, 'w')
    with torch.no_grad():
        for img_file in tqdm(test_df.path):
            predict = [img_file]
            
            for batch in range(len(batch_iter)):
                img_1 = batch_iter.get_batch(batch)
                img_2 = Image.open(os.path.join(img_dir, img_file)).convert('RGB')
                img_2 = to_tensor(img_2)
                img_2 = torch.repeat_interleave(img_2.unsqueeze(0), len(img_1), dim = 0)

                img_1 = img_1.to(device = device)
                img_2 = img_2.to(device = device)
                
                outputs = model(img_1, img_2) # batchx2
                pair_predict = torch.softmax(outputs, dim = 1).argmax(dim = 1)
                predict.extend(pair_predict.cpu().tolist())
            
            # predict = np.array(predict).reshape(-1, 5)
            # max_value = (predict == 0).sum(axis = 1)
            logger.record(predict)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--device', type=str, help='device')
    parser.add_argument('--test-file', type=str, help='path to tests file')
    parser.add_argument('--support-set', type=str, help='path to support file')
    parser.add_argument('--batch-size', type=int, help='batch size')
    parser.add_argument('--out-file', type=str, help='path to out file')
    parser.add_argument('--img-dir', type=str, help='path to image')
    
    args = parser.parse_args()

    cf = utils.Config(args.config)
    cf.device = args.device
    img_dir = args.img_dir

    test_classify(cf, args.test_file, args.support_set, args.batch_size, args.out_file)