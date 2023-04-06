import torch.nn as nn
import torch.optim as optim
import datetime
import copy
import numpy as np
from unicorn.utils.utils import make_cuda, save_model
from unicorn.utils import param
from .evaluate import evaluate_moe, evaluate_wo_moe

def train_moe(args, encoder, moelayer, classifiers,
            train_data_loaders, valid_data_loaders=None, metrics=None, need_save_model=True):
    
    # setup criterion and optimizer
    optimizer0 = optim.Adam(list(encoder.parameters()),
                           lr=args.c_learning_rate)
    
    optimizerm = optim.Adam(list(moelayer.parameters()),
                           lr=args.c_learning_rate)
    
    optimizers = optim.Adam(list(classifiers.parameters()),
                           lr=args.c_learning_rate)              
    CELoss = nn.CrossEntropyLoss()

    start = datetime.datetime.now()
    if valid_data_loaders:
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

        if valid_data_loaders:
            avg_valid = []
            for k in range(len(valid_data_loaders)):
                print("valid  datasets : ",k+1)
                f1,recall,acc = evaluate_moe(encoder, moelayer, classifiers, valid_data_loaders[k], args=args, all=1)
                if metrics[k]=='f1':
                    avg_valid.append(f1)
                if metrics[k]=='recall':
                    avg_valid.append(recall)
                if metrics[k]=='acc' or metrics[k]=='hit':
                    avg_valid.append(acc)
            if np.mean(avg_valid) > bestf1:
                print("best epoch number: ",epoch)
                bestf1 = np.mean(avg_valid)

                best_encoder = copy.deepcopy(encoder)
                best_moelayer = copy.deepcopy(moelayer)
                best_classifiers = copy.deepcopy(classifiers)
        
    
    if not valid_data_loaders:
        best_encoder = copy.deepcopy(encoder)
        best_moelayer = copy.deepcopy(moelayer)
        best_classifiers = copy.deepcopy(classifiers)

    if need_save_model:
        print("save model")
        save_model(args, best_encoder, args.modelname+"_"+param.encoder_path)
        save_model(args, best_moelayer, args.modelname+"_"+param.moe_path)
        save_model(args, best_classifiers, args.modelname+"_"+param.cls_path)
    
    end = datetime.datetime.now()
    print("Time: ",end-start)
    return best_encoder, best_moelayer, best_classifiers

def train_wo_moe(args, encoder, classifiers,
            train_data_loaders, valid_data_loaders=None, metrics=None, need_save_model=True, draw=True):
    
    # setup criterion and optimizer
    optimizer0 = optim.Adam(list(encoder.parameters()),
                           lr=args.c_learning_rate)
    
    optimizers = optim.Adam(list(classifiers.parameters()),
                           lr=args.c_learning_rate)              
    CELoss = nn.CrossEntropyLoss()

    start = datetime.datetime.now()
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
        
        if valid_data_loaders != None:
            avg_valid = []
            for k in range(len(valid_data_loaders)):
                print("valid  datasets : ",k+1)
                f1,recall,acc = evaluate_wo_moe(encoder, classifiers, valid_data_loaders[k],args=args,all=1)
                if metrics[k]=='f1':
                    avg_valid.append(f1)
                if metrics[k]=='recall':
                    avg_valid.append(recall)
                if metrics[k]=='acc' or metrics[k]=='hit':
                    avg_valid.append(acc)
            if np.mean(avg_valid) > bestf1:
                print("best epoch number: ",epoch)
                bestf1 = np.mean(avg_valid)

                best_encoder = copy.deepcopy(encoder)
                best_classifiers = copy.deepcopy(classifiers)

    if not valid_data_loaders:
        best_encoder = copy.deepcopy(encoder)
        best_classifiers = copy.deepcopy(classifiers)

    if need_save_model:
        print("save model")
        save_model(args, best_encoder, args.modelname+"_"+param.encoder_path)
        save_model(args, best_classifiers, args.modelname+"_"+param.cls_path)
    
    end = datetime.datetime.now()
    print("Time: ",end-start)
    return best_encoder, best_classifiers


