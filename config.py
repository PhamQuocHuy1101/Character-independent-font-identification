device = 'cpu'

model = {
    'n_class': 3,
    'backbone': 'resnet18',
    'drop': 0.12,
    'freeze': False,
    'checkpoint': './checkpoint/resnet18.pt',
    'beta': 0.5,
    'block_expansion': 128
}

model_e = {
    'n_class': 2,
    'backbone': 'b0',
    'drop': 0.2,
    'freeze': False,
    'checkpoint': './checkpoint/efficientNetB0.pt'
}

optim = {
    'lr': 0.001,
    'step': 5,
    'reduce_rate': 0.1,
    'batch_size': 32,
    'n_epoch': 15,
    'continue_training': False
}

data = {
    'dir_path': './data/font_images',
    'train_csv': './data/train.csv',
    'val_csv': './data/val.csv',
    'test_csv': './data/test.csv',
}
