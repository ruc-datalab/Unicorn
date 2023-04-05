import torch
import torch.nn as nn
import json
from unicorn.utils.utils import make_cuda
from unicorn.dataprocess import dataformat


def evaluate_moe(encoder, moelayer, classifier, data_loader, args=None, index=-1, flag=None, write=False, prob_name=None, all=None):
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

def evaluate_wo_moe(encoder, classifier, data_loader, args=None, flag=None, write=False, prob_name=None, all=None):
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

