from network import font_model
import importlib

def load_model(cf):
    # font_model = importlib.import_module('network.font_model')
    model_name = getattr(font_model, cf['name'])
    model = model_name(**cf['args'])
    return model