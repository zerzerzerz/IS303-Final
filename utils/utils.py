import random 
import json
import datetime
import numpy as np
import random
import os
import torch

def load_json(path):
    with open(path,'r') as f:
        res = json.load(f)
    return res


def save_json(obj, path:str):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(obj, f, indent=4)


def setup_seed(seed = 3407):
     torch.manual_seed(seed)
     np.random.seed(seed)
     random.seed(seed)


def get_datetime():
    time1 = datetime.datetime.now()
    time2 = datetime.datetime.strftime(time1,'%Y-%m-%d %H:%M:%S')
    return str(time2)


class Logger():
    def __init__(self,log_file_path) -> None:
        self.path = log_file_path
        with open(self.path,'w') as f:
            f.write(get_datetime() + "\n")
            print(get_datetime())
        return
    
    def log(self,content):
        with open(self.path,'a') as f:
            f.write(content + "\n")
            print(content)
        return

def mkdir(dir):
    if os.path.isdir(dir):
        pass
    else:
        os.makedirs(dir)
    return True