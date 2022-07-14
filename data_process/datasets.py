import os
import shutil
import random
import csv
import json
import datetime

csv.field_size_limit(500 * 1024 * 1024)
random.seed(10)

def open_csv(args, file_path):
    f = open(file_path, "r") 
    reader = csv.reader(f)
    lines = []
    zero = False
    for line in reader:
        lines.append(line)
        if line[2] == 0 or line[2] == '0':
            zero = True
    f.close()
    if zero:
        return lines
    return lines[:min(len(lines),args.num_data)]

def save_to_csv(path,data):
    f = open(path, "w")
    writer = csv.writer(f)
    for i in data:
        writer.writerow(i)
    f.close()

def calculate_hits_k(data, prob):
    n = len(data)
    source = ""
    source_num = 0
    tmp = []
    k1 = 0
    k10 = 0
    for i in range(n):
        if data[i][0] == source:
            tmp.append([data[i][2], prob[i]])
        else:
            if tmp:
                tmp = sorted(tmp, key = lambda x:x[1], reverse = True)
                if tmp[0][0] == 1:
                    k1 += 1
                if 1 in [x[0] for x in tmp[:min(10,len(tmp))]]:
                    k10 += 1
            source_num += 1
            source = data[i][0]
            tmp = [ [data[i][2], prob[i]] ]
    print("source_num : ", source_num)
    print("Hit@1 : ", k1/source_num)
    print("Hit@10 : ", k10/source_num)

