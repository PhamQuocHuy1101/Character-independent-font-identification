from utils import Config
from network import load_model

cf = Config('./config/1flow.yaml')
model = load_model(cf.model)
print(model)