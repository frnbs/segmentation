import importlib
from models.base_model import BaseModel
import sys
from utils.colors_text import bcolors


def find_model_using_name(model_name):
    model_filename = "models.{}_model".format(str(model_name.split('_')[0]))
    try:
        modellib = importlib.import_module(model_filename)
    except:
        print("{}Cannot find file {}.py{}".format(bcolors.FAIL, model_filename, bcolors.ENDC))
        print("{}Closing!!! {}".format(bcolors.FAIL, bcolors.ENDC))
        sys.exit()

    model = None
    target_model_name = model_name.split('_')[0] + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In {}.py, there should be a subclass of BaseModel with class name that matches {} in lowercase.".format(
            model_filename.split('_')[0], target_model_name))
        exit(0)

    return model


def create_model(opt):
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance
