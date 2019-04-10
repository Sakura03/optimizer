import shutil, sys, os, torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from os.path import join, split, abspath, dirname, isfile, isdir
import yaml
import math
import numpy as np

def load_yaml(yaml_file):
    assert isfile(yaml_file), "File %s does'nt exist!" % yaml_file
    return yaml.load(open(yaml_file))

def merge_config(args, yaml_config):
    # data
    if hasattr(args, "data") and args.data and args.data != yaml_config["DATA"]["DIR"]:
        yaml_config["DATA"]["DIR"] = args.data

    if hasattr(args, "bs") and args.bs and args.bs != yaml_config["DATA"]["BS"]:
        yaml_config["DATA"]["BS"] = args.bs

    if hasattr(args, "num_classes") and args.num_classes and args.num_classes != yaml_config["DATA"]["NUM_CLASSES"]:
        yaml_config["DATA"]["NUM_CLASSES"] = args.num_classes

    # model
    if hasattr(args, "model") and args.model and args.model != yaml_config["MODEL"]["MODEL"]:
        yaml_config["MODEL"]["MODEL"] = args.model


    # Optimizer
    if hasattr(args, "lr") and args.lr and args.lr != yaml_config["OPTIMIZER"]["LR"]:
        yaml_config["OPTIMIZER"]["LR"] = args.lr
    
    if hasattr(args, "epochs") and args.epochs and args.epochs != yaml_config["OPTIMIZER"]["EPOCHS"]:
        yaml_config["OPTIMIZER"]["EPOCHS"] = args.epochs

    # CUDA
    if hasattr(args, "gpu") and args.gpu and args.gpu != yaml_config["CUDA"]["GPU_ID"]:
        yaml_config["CUDA"]["GPU_ID"] = args.gpu
    
    if hasattr(args, "visport") and args.visport != yaml_config["VISDOM"]["PORT"]:
        yaml_config["VISDOM"]["PORT"] = args.visport

    # MISC
    if yaml_config["MISC"]["TMP"] == "" or yaml_config["MISC"]["TMP"] is None:
        yaml_config["MISC"]["TMP"] = yaml_config["DATA"]["DATASET"] + "-" + yaml_config["MODEL"]["MODEL"]
        yaml_config["MISC"]["TMP"] = join("tmp", yaml_config["MISC"]["TMP"])

    if hasattr(args, "tmp") and args.tmp and args.tmp != yaml_config["MISC"]["TMP"]:
        yaml_config["MISC"]["TMP"] = args.tmp

    return yaml_config

class DayHourMinute(object):
  
  def __init__(self, seconds):
      
      self.days = int(seconds // 86400)
      self.hours = int((seconds- (self.days * 86400)) // 3600)
      self.minutes = int((seconds - self.days * 86400 - self.hours * 3600) // 60)

def get_lr(epoch, base_lr, warmup_epochs=5, warmup_start_lr=0.001):
    lr = 0
    if epoch < warmup_epochs:
        lr = ((base_lr - warmup_start_lr) / warmup_epochs) * epoch
    else:
        lr = base_lr * (0.1 ** ((epoch-warmup_epochs) // 30))
    return lr

class LRScheduler(object):

    def __init__(self, base_lr, max_epochs, iters_per_epoch, gamma=0.1, warmup_epochs=5, warmup_start_lr=0.001):

        self.base_lr = base_lr
        self.max_epochs = max_epochs
        self.iters_per_epoch = iters_per_epoch
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr

        self._it = 0


    def step(self, step=1):

        self._it += step

def set_lr(optimizer, lr):

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, path, epoch, milestones, filename="checkpoint.pth"):

    torch.save(state, join(path, filename))
    if is_best:
        shutil.copyfile(join(path, filename), join(path, 'model_best.pth'))
    if epoch in milestones:
        shutil.copyfile(join(path, filename), join(path, 'epoch%d_checkpoint.pth'%epoch))