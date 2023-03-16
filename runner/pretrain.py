import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import csv
import os
import math
import sys
import datetime
import sys
from itertools import zip_longest
sys.path.append("../")
from utils.utils import make_cuda, save_model
import param
import copy
import numpy as np
import dataformat
import json
import scipy
import collections

def train_multi_moe_1cls_new(args, encoder, moelayer, classifiers,
            train_data_loaders,test_data_loaders,valid_data_loaders=None,need_save_model=True,draw=True):
    
    # setup criterion and optimizer
    optimizer0 = optim.Adam(list(encoder.parameters()),
                           lr=args.c_learning_rate)
    
    optimizerm = optim.Adam(list(moelayer.parameters()),
                           lr=args.c_learning_rate)
    
    optimizers = optim.Adam(list(classifiers.parameters()),
                           lr=args.c_learning_rate)              
    CELoss = nn.CrossEntropyLoss()

    start = datetime.datetime.now()
    bestaverage = 0.0
    if valid_data_loaders != None:
        bestf1 = 0.0
        best_encoder = copy.deepcopy(encoder)
        best_moelayer = copy.deepcopy(moelayer)
        best_classifiers = copy.deepcopy(classifiers)

    for epoch in range(args.pre_epochs):
        encoder.train()
        moelayer.train()
        classifiers.train()
        for i in range(len(train_data_loaders)):
            for step, pair in enumerate(train_data_loaders[i]):
                values1 = make_cuda(pair[0])
                mask1 = make_cuda(pair[1])
                segment1 = make_cuda(pair[2])

                labels = make_cuda(pair[3])

                # zero gradients for optimizer
                optimizer0.zero_grad()
                optimizerm.zero_grad()
                optimizers.zero_grad()

                # compute loss for discriminator
                if args.model in ['distilbert','distilroberta']:
                    feat = encoder(values1,mask1)
                else:
                    feat = encoder(values1, mask1, segment1)
                if args.load_balance:
                    moeoutput,balanceloss,entroloss,_ = moelayer(feat)
                else:
                    moeoutput,_ = moelayer(feat)
                preds = classifiers(moeoutput)
                cls_loss = CELoss(preds, labels)
                if args.load_balance:
                    loss = cls_loss + args.balance_loss*balanceloss + args.entroloss*entroloss
                else:
                    loss = cls_loss
                # optimize source classifier
                loss.backward()
                optimizers.step()
                optimizerm.step()
                optimizer0.step()

                # print step info
                if args.load_balance:
                    if (step + 1) % args.pre_log_step == 0:
                        print("Epoch [%.2d/%.2d] Step [%.3d]: cls_loss=%.4f %4f %4f"
                            % (epoch + 1,
                                args.pre_epochs,
                                step + 1,
                                cls_loss.item(),
                                balanceloss.item(),
                                entroloss.item()))
                else:
                    if (step + 1) % args.pre_log_step == 0:
                        print("Epoch [%.2d/%.2d] Step [%.3d]: cls_loss=%.4f"
                            % (epoch + 1,
                                args.pre_epochs,
                                step + 1,
                                cls_loss.item()))

        testaverage = 0
        metrics = ['hit','hit','f1','f1','f1','f1','f1','recall','recall','acc','acc','acc','acc','f1','f1',
                   'f1','f1','f1','f1','f1']
        for k in range(2,len(test_data_loaders)):
            print("test  datasets : ",k+1)
            f1,recall,acc = evaluate_moe_new(encoder, moelayer, classifiers, test_data_loaders[k],args=args,all=1)
            if metrics[k]=='f1':
                testaverage += f1
            if metrics[k]=='recall':
                testaverage += recall
            if metrics[k]=='acc':
                testaverage += acc
        testaverage = testaverage/(len(test_data_loaders)-2)
        print("testaverage:::",testaverage)
        
        if valid_data_loaders != None:
            f1_valid = []
            for k in range(0,len(valid_data_loaders)):
                print("valid  datasets : ",k+1)
                f1,recall,acc = evaluate_moe_new(encoder, moelayer, classifiers, valid_data_loaders[k],args=args,all=1)
                if metrics[k]=='f1':
                    f1_valid.append(f1)
                if metrics[k]=='recall':
                    f1_valid.append(recall)
                if metrics[k]=='acc':
                    f1_valid.append(acc)
            if np.mean(f1_valid) > bestf1:
                print("best epoch number: ",epoch)
                bestf1 = np.mean(f1_valid)

                best_encoder = copy.deepcopy(encoder)
                best_moelayer = copy.deepcopy(moelayer)
                best_classifiers = copy.deepcopy(classifiers)
                best_average = testaverage
        
    
    if not valid_data_loaders:
        best_encoder = copy.deepcopy(encoder)
        best_moelayer = copy.deepcopy(moelayer)
        best_classifiers = copy.deepcopy(classifiers)

    if need_save_model:
        print("save model")
        save_model(args, best_encoder, args.namef+"_"+param.encoder_path)
        save_model(args, best_moelayer, args.namef+"_"+param.moe_path)
        save_model(args, best_classifiers, args.namef+"_"+param.cls_path)
    
    end = datetime.datetime.now()
    print("Time: ",end-start)
    return best_encoder, best_moelayer, best_classifiers

def train_multi_1cls_new(args, encoder, classifiers,
            train_data_loaders,test_data_loaders,valid_data_loaders=None,need_save_model=True,draw=True):
    
    # setup criterion and optimizer
    optimizer0 = optim.Adam(list(encoder.parameters()),
                           lr=args.c_learning_rate)
    
    optimizers = optim.Adam(list(classifiers.parameters()),
                           lr=args.c_learning_rate)              
    CELoss = nn.CrossEntropyLoss()

    start = datetime.datetime.now()
    bestaverage = 0.0
    if valid_data_loaders != None:
        bestf1 = 0.0
        best_encoder = copy.deepcopy(encoder)
        best_classifiers = copy.deepcopy(classifiers)

    for epoch in range(args.pre_epochs):
        encoder.train()
        classifiers.train()
        for i in range(len(train_data_loaders)):
            for step, pair in enumerate(train_data_loaders[i]):
                values1 = make_cuda(pair[0])
                mask1 = make_cuda(pair[1])
                segment1 = make_cuda(pair[2])

                labels = make_cuda(pair[3])

                # zero gradients for optimizer
                optimizer0.zero_grad()
                optimizers.zero_grad()

                # compute loss for discriminator
                if args.model in ['distilbert','distilroberta']:
                    feat = encoder(values1,mask1)
                else:
                    feat = encoder(values1, mask1, segment1)
                preds = classifiers(feat)
                cls_loss = CELoss(preds, labels)
                loss = cls_loss
                # optimize source classifier
                loss.backward()
                optimizers.step()
                optimizer0.step()

                # print step info
                if (step + 1) % args.pre_log_step == 0:
                    print("Epoch [%.2d/%.2d] Step [%.3d]: cls_loss=%.4f"
                        % (epoch + 1,
                            args.pre_epochs,
                            step + 1,
                            cls_loss.item()))
        testaverage = 0
        metrics = ['hit','hit','f1','f1','f1','f1','f1','recall','recall','acc','acc','acc','acc','f1','f1',
                   'f1','f1','f1','f1','f1']
        for k in range(2,len(test_data_loaders)):
            print("test  datasets : ",k+1)
            f1,recall,acc = evaluate_new(encoder, classifiers, test_data_loaders[k],args=args,all=1)
            if metrics[k]=='f1':
                testaverage += f1
                print('f1',f1)
            if metrics[k]=='recall':
                testaverage += recall
                print('recall',recall)
            if metrics[k]=='acc':
                testaverage += acc
                print('acc',acc)
        testaverage = testaverage/(len(test_data_loaders)-2)
        print("testaverage:::",testaverage)
        
        if valid_data_loaders != None:
            f1_valid = []
            for k in range(0,len(valid_data_loaders)):
                print("valid  datasets : ",k+1)
                f1,recall,acc = evaluate_new(encoder, classifiers, valid_data_loaders[k],args=args,all=1)
                if metrics[k]=='f1':
                    f1_valid.append(f1)
                if metrics[k]=='recall':
                    f1_valid.append(recall)
                if metrics[k]=='acc':
                    f1_valid.append(acc)
            if np.mean(f1_valid) > bestf1:
                print("best epoch number: ",epoch)
                bestf1 = np.mean(f1_valid)

                best_encoder = copy.deepcopy(encoder)
                best_classifiers = copy.deepcopy(classifiers)
                best_average = testaverage
        

    if need_save_model:
        print("save model")
        save_model(args, best_encoder, args.namef+"_"+param.encoder_path)
        save_model(args, best_classifiers, args.namef+"_"+param.cls_path)
    
    end = datetime.datetime.now()
    print("Time: ",end-start)
    return best_encoder, best_classifiers

def train_multi_moe_1cls_new_zero(args, encoder, moelayer, classifiers,
            train_data_loaders,test_data_loaders=None,valid_data_loaders=None,need_save_model=True,draw=True):
    
    # setup criterion and optimizer
    optimizer0 = optim.Adam(list(encoder.parameters()),
                           lr=args.c_learning_rate)
    
    optimizerm = optim.Adam(list(moelayer.parameters()),
                           lr=args.c_learning_rate)
    
    optimizers = optim.Adam(list(classifiers.parameters()),
                           lr=args.c_learning_rate)              
    CELoss = nn.CrossEntropyLoss()

    start = datetime.datetime.now()
    if valid_data_loaders != None:
        bestf1 = 0.0
        best_encoder = copy.deepcopy(encoder)
        best_moelayer = copy.deepcopy(moelayer)
        best_classifiers = copy.deepcopy(classifiers)

    for epoch in range(args.pre_epochs):
        encoder.train()
        moelayer.train()
        classifiers.train()
        for i in range(len(train_data_loaders)):
            for step, pair in enumerate(train_data_loaders[i]):
                values1 = make_cuda(pair[0])
                mask1 = make_cuda(pair[1])
                segment1 = make_cuda(pair[2])

                labels = make_cuda(pair[3])

                # zero gradients for optimizer
                optimizer0.zero_grad()
                optimizerm.zero_grad()
                optimizers.zero_grad()

                # compute loss for discriminator
                if args.model in ['distilbert','distilroberta']:
                    feat = encoder(values1,mask1)
                else:
                    feat = encoder(values1, mask1, segment1)
                if args.load_balance:
                    moeoutput,balanceloss,entroloss,_ = moelayer(feat)
                else:
                    moeoutput,_ = moelayer(feat)
                preds = classifiers(moeoutput)
                cls_loss = CELoss(preds, labels)
                if args.load_balance:
                    loss = cls_loss + 0.01*balanceloss
                else:
                    loss = cls_loss
                # optimize source classifier
                loss.backward()
                optimizers.step()
                optimizerm.step()
                optimizer0.step()

                # print step info
                if (step + 1) % args.pre_log_step == 0:
                    print("Epoch [%.2d/%.2d] Step [%.3d]: cls_loss=%.4f"
                        % (epoch + 1,
                            args.pre_epochs,
                            step + 1,
                            cls_loss.item()))
    
        if valid_data_loaders != None:
            f1_valid = []
            for k in range(0,len(valid_data_loaders)):
                print("valid  datasets : ",k+1)
                f1_valid.append( evaluate_moe_new(encoder, moelayer, classifiers, valid_data_loaders[k],args=args) )
            if np.mean(f1_valid) > bestf1:
                print("best epoch number: ",epoch)
                bestf1 = np.mean(f1_valid)

                best_encoder = copy.deepcopy(encoder)
                best_moelayer = copy.deepcopy(moelayer)
                best_classifiers = copy.deepcopy(classifiers)
        
    if 1:
        print("save model")
        save_model(args, best_encoder, args.namef+"_"+param.encoder_path)
        save_model(args, best_moelayer, args.namef+"_"+param.moe_path)
        save_model(args, best_classifiers, args.namef+"_"+param.cls_path)
    
    end = datetime.datetime.now()
    print("Time: ",end-start)
    return best_encoder, best_moelayer, best_classifiers

def evaluate_moe_new(encoder, moelayer,classifier, data_loader,args=None,index=-1,flag=None,discriminator=None,exp_idx=None,write=False, prob_name=None, all=None):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    moelayer.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0
    tp = 0
    fp = 0
    p = 0
    need_preds = []
    # set loss function
    criterion = nn.CrossEntropyLoss()
    count = 0
    # evaluate network
    confidences = {}
    probility = {}
    averagegateweight = torch.Tensor([0 for _ in range(args.expertsnum)]).cuda()
    for (reviews, mask,segment, labels,exm_id,task_id) in data_loader:
        reviews = make_cuda(reviews)
        mask = make_cuda(mask)
        segment = make_cuda(segment)
        labels = make_cuda(labels)
        truelen = torch.sum(mask, dim=1)
        
        with torch.no_grad():
            if args.model in ['distilbert','distilroberta']:
                feat = encoder(reviews,mask)
            else:
                feat = encoder(reviews, mask, segment)
            if index==-1:
                if args.load_balance:
                    moeoutput,balanceloss,_,gateweights = moelayer(feat)
                    averagegateweight += gateweights
                else:
                    moeoutput,gateweights = moelayer(feat)
                    averagegateweight += gateweights
            else:
                moeoutput,_ = moelayer(feat,gate_idx=index)
            preds = classifier(moeoutput)
            if flag == "get_prob":
                for i in range(len(preds)):
                    probility[exm_id[i].item()] = preds[i][1].item()
            if write:
                for i in range(len(preds)):
                    confidences[exm_id[i].item()] = abs(preds[i][0].item()-preds[i][1].item())

        loss += criterion(preds, labels).item()
        pred_cls = preds.data.max(1)[1]

        acc += pred_cls.eq(labels.data).cpu().sum().item()
        for i in range(len(labels)):
            if labels[i] == 1:
                p += 1
                if pred_cls[i] == 1:
                    tp += 1
            else:
                if pred_cls[i] == 1:
                    fp += 1
    
    print(averagegateweight)
    print("gatedistribution:",averagegateweight/torch.sum(averagegateweight))
    if flag == "get_prob" and prob_name:
        with open(prob_name, 'w') as f_obj:  
            json.dump(probility, f_obj)
    if write:
        id_etro = sorted(confidences.items(),  key=lambda d: d[1], reverse=False)
        select = [id_[0] for id_ in id_etro[:200]]
        with open(dataformat.deepmatcher_data[args.datasetkey]+str(args.labelnum)+'selecttrain.json','r',encoding='utf8') as fw:
            alreadyselect = json.load(fw)
        with open(dataformat.deepmatcher_data[args.datasetkey]+str(args.labelnum)+'remaintrain.json','r',encoding='utf8') as fw:
            waitselect = json.load(fw)
        addselect = [waitselect[i] for i in range(len(waitselect)) if i in select]
        remainselect = [waitselect[i] for i in range(len(waitselect)) if i not in select]
        newselect = alreadyselect + addselect
        with open(dataformat.deepmatcher_data[args.datasetkey]+str(args.labelnum+200)+"selecttrain.json","w") as f:
            json.dump(newselect,f)
        with open(dataformat.deepmatcher_data[args.datasetkey]+str(args.labelnum+200)+"remaintrain.json","w") as f:
            json.dump(remainselect,f)

    
    div_safe = 0.000001
    print("p",p)
    print("tp",tp)
    print("fp",fp)
    recall = tp/(p+div_safe)
    
    precision = tp/(tp+fp+div_safe)
    f1 = 2*recall*precision/(recall + precision + div_safe)
    print("recall",recall)
    print("precision",precision)
    print("f1",f1)
    print(len(data_loader))

    acc /= len(data_loader.dataset)
    print("Avg Loss = %.4f, Avg Accuracy = %.4f" % (loss, acc))
    print("====================================================")
    if flag == "get_prob":
        return probility
    if all == 1:
        return f1, recall, acc
    return f1

def evaluate_new(encoder, classifier, data_loader,args=None,index=-1,flag=None,discriminator=None,exp_idx=None,write=False, prob_name=None, all=None):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0
    tp = 0
    fp = 0
    p = 0
    need_preds = []
    # set loss function
    criterion = nn.CrossEntropyLoss()
    count = 0
    # evaluate network
    confidences = {}
    probility = {}
    for (reviews, mask,segment, labels,exm_id,_) in data_loader:
        reviews = make_cuda(reviews)
        mask = make_cuda(mask)
        segment = make_cuda(segment)
        labels = make_cuda(labels)
        truelen = torch.sum(mask, dim=1)
        
        with torch.no_grad():
            if args.model in ['distilbert','distilroberta']:
                feat = encoder(reviews,mask)
            else:
                feat = encoder(reviews, mask, segment)
            preds = classifier(feat)
            if flag == "get_prob":
                for i in range(len(preds)):
                    probility[exm_id[i].item()] = preds[i][1].item()
            if write:
                for i in range(len(preds)):
                    confidences[exm_id[i].item()] = abs(preds[i][0].item()-preds[i][1].item())

        loss += criterion(preds, labels).item()
        pred_cls = preds.data.max(1)[1]

        acc += pred_cls.eq(labels.data).cpu().sum().item()
        for i in range(len(labels)):
            if labels[i] == 1:
                p += 1
                if pred_cls[i] == 1:
                    tp += 1
            else:
                if pred_cls[i] == 1:
                    fp += 1

    if flag == "get_prob" and prob_name:
        with open(prob_name, 'w') as f_obj:  
            json.dump(probility, f_obj)
    if write:
        id_etro = sorted(confidences.items(),  key=lambda d: d[1], reverse=False)
        select = [id_[0] for id_ in id_etro[:200]]
        with open(dataformat.deepmatcher_data[args.datasetkey]+str(args.labelnum)+'selecttrain.json','r',encoding='utf8') as fw:
            alreadyselect = json.load(fw)
        with open(dataformat.deepmatcher_data[args.datasetkey]+str(args.labelnum)+'remaintrain.json','r',encoding='utf8') as fw:
            waitselect = json.load(fw)
        addselect = [waitselect[i] for i in range(len(waitselect)) if i in select]
        remainselect = [waitselect[i] for i in range(len(waitselect)) if i not in select]
        newselect = alreadyselect + addselect
        with open(dataformat.deepmatcher_data[args.datasetkey]+str(args.labelnum+200)+"selecttrain.json","w") as f:
            json.dump(newselect,f)
        with open(dataformat.deepmatcher_data[args.datasetkey]+str(args.labelnum+200)+"remaintrain.json","w") as f:
            json.dump(remainselect,f)


    div_safe = 0.000001
    print("p",p)
    print("tp",tp)
    print("fp",fp)
    recall = tp/(p+div_safe)
    
    precision = tp/(tp+fp+div_safe)
    f1 = 2*recall*precision/(recall + precision + div_safe)
    print("recall",recall)
    print("precision",precision)
    print("f1",f1)
    print(len(data_loader))

    acc /= len(data_loader.dataset)
    print("Avg Loss = %.4f, Avg Accuracy = %.4f" % (loss, acc))
    print("====================================================")
    if flag == "get_prob":
        return probility
    if all == 1:
        return f1, recall, acc
    return f1

