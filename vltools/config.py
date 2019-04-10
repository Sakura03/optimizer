import yaml
import os
from os.path import join, split, abspath, dirname, isfile, isdir


def load_yaml(yaml_file):
    assert isfile(yaml_file), "File %s does'nt exist!" % yaml_file
    return yaml.load(open(yaml_file))

def merge_config(args, yaml_config):

    if hasattr(args, "tmp") and args.tmp != yaml_config["TMP"]["DIR"]:
        yaml_config["TMP"]["DIR"] = args.tmp

    if hasattr(args, "lr") and args.lr != yaml_config["OPTIMIZER"]["LR"]:
        yaml_config["OPTIMIZER"]["LR"] = args.lr
    
    if hasattr(args, "gpu") and args.gpu != yaml_config["CUDA"]["GPU_ID"]:
        yaml_config["CUDA"]["GPU_ID"] = args.gpu
    
    if hasattr(args, "visport") and args.gpu != yaml_config["VISDOM"]["PORT"]:
        yaml_config["VISDOM"]["PORT"] = args.visport

    return yaml_config

def str2bool(x):

    positives = ["yes", "true", "1", "y"]
    negatives = ["no", "false", "0", "n"]
    x = x.lower()

    if x in positives:
        return True
    elif x in negatives:
        return False
    else:
        raise ValueError("Unknown option %s" % x)
    