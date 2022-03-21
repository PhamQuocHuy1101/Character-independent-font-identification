import argparse
from ast import parse
import torch
from PIL import Image
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from network import load_model
import utils

from data.transform import to_tensor
from data.model_dataset import FontDataset


def test_classify(cf, predict_out_file):
    device = cf.device

    model = load_model(cf.model)
    checkpoint = torch.load(cf.model['checkpoint'], map_location = device)
    checkpoint_model = {k[7:]:v for k, v in checkpoint['model'].items()}
    model.load_state_dict(checkpoint_model)
    model.eval()
    print('load state dict')


    test_df = pd.read_csv(cf.data['test_csv'])
    test_data = FontDataset(cf.data['dir_path'], test_df.sample_1, test_df.sample_2, test_df.labels, to_tensor)

    predict_label = []
    with torch.no_grad():
        test_acc = 0.0, 0.0
        for i in tqdm(range(len(test_data))):
            img_1, img_2, labels = test_data[i]
            img_2 = img_2.to(device = device)
            labels = labels.to(device = device)
            
            outputs = model(img_1.unsqueeze(0), img_2.unsqueeze(0))
            y_predict = torch.softmax(outputs, dim = 1).argmax(dim = 1)
            predict_label.append(y_predict.item())
            
            test_acc += torch.sum(y_predict == labels).item()

        test_acc /= len(test_data)

    with open(predict_out_file, 'w') as f:
        f.write('\t'.join([str(i) for i in predict_label]))

    confusion_matrix(test_df.labels, predict_label)
    print(classification_report(test_df.labels, predict_label))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--device', type=str, help='device')
    parser.add_argument('--out-file', type=str, help='path to out file')
    args = parser.parse_args()

    cf = utils.Config(args.config)
    cf.device = args.device

    test_classify(cf)