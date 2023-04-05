#-*- coding : utf-8 -*-
# coding: unicode_escape
import os
import random
import json
import csv
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from unicorn.utils import param


def read_csv(input_file):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding='gbk', errors='ignore') as f:
      reader = csv.reader(f)
      lines = []
      for line in reader:
        lines.append(line)
      return lines[1:]

def norm(s):
    s = s.replace(","," ")
    return s

def getstr(id, data):
    for x in data:
        if str(id) == str(x[0]):
            return norm(" ".join(x[1:]))

def read_data_from_raw(af, bf, goldf):
    a = read_csv(af)
    b = read_csv(bf)
    g = read_csv(goldf)
    pos = 0
    neg = 0
    res = []
    for x in g:
        x[0],x[1] = int(x[0]), int(x[1])
    for x in g[:2000]:
        lst = getstr(x[0],a)
        rst = getstr(x[1],b)
        res.append([lst, rst, 1])
        pos += 1
        fake = x[1]+1
        while [x[0], fake] in g:
            fake += 1
        lst = getstr(x[0],a)
        rst = getstr(fake,b)
        res.append([lst, rst, 0])
        neg += 1
    print("pos ", pos)
    print("neg ", neg)
    return res

def save_to_csv(filename,data):
    with open(filename, 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        writer.writerows(data)


def get_data(filename,num=-1):
    data = json.load(open(filename,encoding='utf-8'))
    if num!=-1 and num<len(data):
        random.seed(42)
        data = random.sample(data,num)
    return data


def save_json(outname,data):
    with open(outname,'w',encoding='utf-8') as file_obj:
        json.dump(data,file_obj,ensure_ascii=False)


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids,label_id,exm_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.exm_id = exm_id


def CSV2Array(path):
    
    data = pd.read_csv(path, encoding='latin')
    reviews, labels = data.reviews.values.tolist(), data.labels.values.tolist()
    return reviews, labels

def make_cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def init_random_seed(manual_seed):
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_model(args, net, restore=None):
    # restore model weights
    if restore is not None:
        path = os.path.join(param.model_root, restore)
        if os.path.exists(path):
            net.load_state_dict(torch.load(path))
            print("Restore model from: {}".format(os.path.abspath(path)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()
    return net

def save_model(args, net, name):
    folder = param.model_root
    path = os.path.join(folder, name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(net.state_dict(), path)
    print("save pretrained model to: {}".format(path))

def draw_f1_line(datas,path):
    x = [i+1 for i in range(len(datas[0]))]
    plt.title('F1 score')    
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.xlabel('Epoch')   
    plt.ylabel('F1')  
    for i,y in enumerate(datas):
        plt.plot(x, y, marker='o', markersize=3, label="task"+str(i))  
    plt.legend(loc='best',frameon=False)
    plt.show()
    plt.savefig(path)

