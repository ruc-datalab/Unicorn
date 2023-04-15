"""Main script for Unicorn."""
from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer, DebertaTokenizer, XLNetTokenizer, DistilBertTokenizer
import torch

import random
import csv
import argparse

from unicorn.model.encoder import (BertEncoder, MPEncoder, DistilBertEncoder, DistilRobertaEncoder, DebertaBaseEncoder, DebertaLargeEncoder,
                   RobertaEncoder, XLNetEncoder)
from unicorn.model.matcher import  MOEClassifier
from unicorn.model.moe import MoEModule
from unicorn.trainer import pretrain, evaluate
from unicorn.utils.utils import get_data, init_model
from unicorn.dataprocess import predata, dataformat
from unicorn.utils import param

csv.field_size_limit(500 * 1024 * 1024)

def parse_arguments():
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")
    parser.add_argument('--pretrain', default=False, action='store_true',
                        help='Force to pretrain source encoder/moe/classifier')

    parser.add_argument('--seed', type=int, default=42,
                        help="Specify random state")

    parser.add_argument('--train_seed', type=int, default=42,
                        help="Specify random state")

    parser.add_argument('--load', default=False, action='store_true',
                        help="Load saved model")

    parser.add_argument('--model', type=str, default="bert",
                        help="Specify model type")

    parser.add_argument('--max_seq_length', type=int, default=128,
                        help="Specify maximum sequence length")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--clip_value", type=float, default=0.01,
                        help="lower and upper clip value for disc. weights")

    parser.add_argument('--batch_size', type=int, default=32,
                        help="Specify batch size")

    parser.add_argument('--pre_epochs', type=int, default=10,
                        help="Specify the number of epochs for pretrain")
        
    parser.add_argument('--pre_log_step', type=int, default=10,
                        help="Specify log step size for pretrain")

    parser.add_argument('--log_step', type=int, default=10,
                        help="Specify log step size for adaptation")

    parser.add_argument('--c_learning_rate', type=float, default=3e-6,
                        help="Specify lr for training")

    parser.add_argument('--num_cls', type=int, default=5,
                        help="")
    parser.add_argument('--num_tasks', type=int, default=2,
                        help="")

    parser.add_argument('--resample', type=int, default=0,
                        help="")
    parser.add_argument('--modelname', type=str, default="Unicorn",
                        help="Specify saved model name")
    parser.add_argument('--ckpt', type=str, default="",
                        help="Specify loaded model name")
    parser.add_argument('--num_data', type=int, default=1000,
                        help="")
    parser.add_argument('--num_k', type=int, default=2,
                        help="")

    parser.add_argument('--scale', type=float, default=20, 
                    help="Use 20 for cossim, and 1 when you work with unnormalized embeddings with dot product")    
    
    parser.add_argument('--wmoe', type=int, default=1, 
                    help="with or without moe")
    parser.add_argument('--expertsnum', type=int, default=6, 
                    help="number of experts")
    parser.add_argument('--size_output', type=int, default=768,
                        help="encoder output size")
    parser.add_argument('--units', type=int, default=768, 
                    help="number of hidden")

    parser.add_argument('--shuffle', type=int, default=0, help="")
    parser.add_argument('--load_balance', type=int, default=0, help="")
    parser.add_argument('--balance_loss',type=float,default=0.1, help="")
    parser.add_argument('--entroloss',type=float,default=0.1, help="")    


    return parser.parse_args()


args = parse_arguments()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

def main():
    # argument setting
    print("=== Argument Setting ===")
    print("experts",args.expertsnum)
    print("encoder: " + str(args.model))
    print("max_seq_length: " + str(args.max_seq_length))
    print("batch_size: " + str(args.batch_size))
    print("epochs: " + str(args.pre_epochs))
    set_seed(args.train_seed)
    if args.model in ['roberta', 'distilroberta']:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    if args.model == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    if args.model == 'mpnet':
        tokenizer = AutoTokenizer.from_pretrained('all-mpnet-base-v2')
    if args.model == 'deberta_base':
        tokenizer = DebertaTokenizer.from_pretrained('deberta-base')
    if args.model == 'deberta_large':
        tokenizer = DebertaTokenizer.from_pretrained('deberta-large')
    if args.model == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    if args.model == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    if args.model == 'bert':
        encoder = BertEncoder()
    if args.model == 'mpnet':
        encoder = MPEncoder()
    if args.model == 'deberta_base':
        encoder = DebertaBaseEncoder()
    if args.model == 'deberta_large':
        encoder = DebertaLargeEncoder()
    if args.model == 'xlnet':
        encoder = XLNetEncoder()
    if args.model == 'distilroberta':
        encoder = DistilRobertaEncoder()
    if args.model == 'distilbert':
        encoder = DistilBertEncoder()
    if args.model == 'roberta':
        encoder = RobertaEncoder()
    
    classifiers = MOEClassifier(args.units) 

    # one moe layer for selection
    exp = args.expertsnum
    moelayer = MoEModule(args.size_output,args.units,exp,load_balance=args.load_balance)
    
    if args.load:
        encoder = init_model(args, encoder, restore=args.ckpt+"_"+param.encoder_path)
        moelayer = init_model(args, moelayer, restore=args.ckpt+"_"+param.moe_path)
        classifiers = init_model(args, classifiers, restore=args.ckpt+"_"+param.cls_path)
    else:
        encoder = init_model(args, encoder)
        classifiers = init_model(args, classifiers)
        moelayer = init_model(args, moelayer)
    
    metrics = []
    if args.pretrain and (not args.shuffle):
        train_sets = []
        test_sets = []
        valid_sets = []
        limit = 10000
        for key,p in dataformat.entity_alignment_data.items():
            train_sets.append(get_data(p[1]+"train.json",num=limit))
            test_sets.append(get_data(p[1]+"test.json"))
            valid_sets.append(get_data(p[1]+"valid.json",num=limit))
            metrics.append(p[2])
        
        for key,p in dataformat.string_matching_data.items():
            train_sets.append(get_data(p[1]+"train.json",num=limit))
            test_sets.append(get_data(p[1]+"test.json"))
            valid_sets.append(get_data(p[1]+"valid.json",num=limit))
            metrics.append(p[2])
        
        for key,p in dataformat.new_schema_matching_data.items():
            train_sets.append(get_data(p[1]+"train.json",num=limit))
            test_sets.append(get_data(p[1]+"test.json"))
            valid_sets.append(get_data(p[1]+"valid.json",num=limit))
            metrics.append(p[2])

        for key,p in dataformat.ontology_matching_data.items():
            train_sets.append(get_data(p[1]+"train.json",num=limit))
            test_sets.append(get_data(p[1]+"test.json"))
            valid_sets.append(get_data(p[1]+"valid.json",num=limit))
            metrics.append(p[2])
        
        for key,p in dataformat.column_type_data.items():
            train_sets.append(get_data(p[1]+"train.json",num=limit))
            test_sets.append(get_data(p[1]+"test.json"))
            valid_sets.append(get_data(p[1]+"valid.json",num=limit))
            metrics.append(p[2])

        for key,p in dataformat.entity_linking_data.items():
            train_sets.append(get_data(p[1]+"train.json",num=limit))
            test_sets.append(get_data(p[1]+"test.json"))
            valid_sets.append(get_data(p[1]+"valid.json",num=limit))
            metrics.append(p[2])

        for key,p in dataformat.new_deepmatcher_data.items():
            train_sets.append(get_data(p[1]+"train.json",num=limit))
            test_sets.append(get_data(p[1]+"test.json"))
            valid_sets.append(get_data(p[1]+"valid.json",num=limit))
            metrics.append(p[2])

        train_data_loaders = []
        test_data_loaders = []
        valid_data_loaders = []
        if args.model in ['bert','deberta_base','deberta_large','distilbert','mpnet']:
            for i in range(len(train_sets)):
                fea = predata.convert_examples_to_features([ [x[0]+" [SEP] " +x[1]] for x in train_sets[i] ], [int(x[2]) for x in train_sets[i]], args.max_seq_length, tokenizer)
                train_data_loaders.append(predata.convert_fea_to_tensor(fea, args.batch_size, do_train=1))
            for i in range(len(valid_sets)):
                fea = predata.convert_examples_to_features([ [x[0]+" [SEP] " +x[1]] for x in valid_sets[i] ], [int(x[2]) for x in valid_sets[i]], args.max_seq_length, tokenizer)
                valid_data_loaders.append(predata.convert_fea_to_tensor(fea, args.batch_size, do_train=0))
            for i in range(len(test_sets)):
                fea = predata.convert_examples_to_features([ [x[0]+" [SEP] "+x[1]] for x in test_sets[i] ], [int(x[2]) for x in test_sets[i]], args.max_seq_length, tokenizer)
                test_data_loaders.append(predata.convert_fea_to_tensor(fea, args.batch_size, do_train=0))
        if args.model in ['roberta','distilroberta']:
            for i in range(len(train_sets)):
                fea = predata.convert_examples_to_features_roberta([ [x[0]+" [SEP] "+x[1]] for x in train_sets[i] ], [int(x[2]) for x in train_sets[i]], args.max_seq_length, tokenizer)
                train_data_loaders.append(predata.convert_fea_to_tensor(fea, args.batch_size, do_train=1))
            for i in range(len(valid_sets)):
                fea = predata.convert_examples_to_features_roberta([ [x[0]+" [SEP] "+x[1]] for x in valid_sets[i] ], [int(x[2]) for x in valid_sets[i]], args.max_seq_length, tokenizer)
                valid_data_loaders.append(predata.convert_fea_to_tensor(fea, args.batch_size, do_train=0))
            for i in range(len(test_sets)):
                fea = predata.convert_examples_to_features_roberta([ [x[0]+" [SEP] "+x[1]] for x in test_sets[i] ], [int(x[2]) for x in test_sets[i]], args.max_seq_length, tokenizer)
                test_data_loaders.append(predata.convert_fea_to_tensor(fea, args.batch_size, do_train=0))
        if args.model=='xlnet':
            for i in range(len(train_sets)):
                fea = predata.convert_examples_to_features([ [x[0]+" [SEP] "+x[1]] for x in train_sets[i] ], [int(x[2]) for x in train_sets[i]], args.max_seq_length, tokenizer, cls_token='<cls>', sep_token='<sep>')
                train_data_loaders.append(predata.convert_fea_to_tensor(fea, args.batch_size, do_train=1))
            for i in range(len(valid_sets)):
                fea = predata.convert_examples_to_features([ [x[0]+" [SEP] "+x[1]] for x in valid_sets[i] ], [int(x[2]) for x in valid_sets[i]], args.max_seq_length, tokenizer, cls_token='<cls>', sep_token='<sep>')
                valid_data_loaders.append(predata.convert_fea_to_tensor(fea, args.batch_size, do_train=0))
            for i in range(len(test_sets)):
                fea = predata.convert_examples_to_features([ [x[0]+" [SEP] "+x[1]] for x in test_sets[i] ], [int(x[2]) for x in test_sets[i]], args.max_seq_length, tokenizer, cls_token='<cls>', sep_token='<sep>')
                test_data_loaders.append(predata.convert_fea_to_tensor(fea, args.batch_size, do_train=0))

        print("train datasets num: ",len(train_data_loaders))
        print("test datasets num: ",len(test_data_loaders))
        print("valid datasets num: ",len(valid_data_loaders))
        encoder, moelayer, classifiers = pretrain.train_moe(args, encoder, moelayer, classifiers, train_data_loaders, valid_data_loaders, metrics)
    
    if args.pretrain and args.shuffle:
        train_sets = []
        test_sets = []
        valid_sets = []
        limit = 10000
        for key,p in dataformat.entity_alignment_data.items():
            traindata = get_data(p[1]+"train.json",num=limit)
            train_sets.extend(traindata)
            test_sets.append(get_data(p[1]+"test.json"))
            valid_sets.append(get_data(p[1]+"valid.json",num=limit))
            metrics.append(p[2])
                    
        for key,p in dataformat.string_matching_data.items():
            traindata = get_data(p[1]+"train.json",num=limit)
            train_sets.extend(traindata)
            test_sets.append(get_data(p[1]+"test.json"))
            valid_sets.append(get_data(p[1]+"valid.json",num=limit))
            metrics.append(p[2])
        
        for key,p in dataformat.new_schema_matching_data.items():
            traindata = get_data(p[1]+"train.json",num=limit)
            train_sets.extend(traindata)
            test_sets.append(get_data(p[1]+"test.json"))
            valid_sets.append(get_data(p[1]+"valid.json",num=limit))
            metrics.append(p[2])

        for key,p in dataformat.ontology_matching_data.items():
            traindata = get_data(p[1]+"train.json",num=limit)
            train_sets.extend(traindata)
            test_sets.append(get_data(p[1]+"test.json"))
            valid_sets.append(get_data(p[1]+"valid.json",num=limit))
            metrics.append(p[2])

        for key,p in dataformat.column_type_data.items():
            traindata = get_data(p[1]+"train.json",num=limit)
            train_sets.extend(traindata)
            test_sets.append(get_data(p[1]+"test.json"))
            valid_sets.append(get_data(p[1]+"valid.json",num=limit))
            metrics.append(p[2])

        for key,p in dataformat.entity_linking_data.items():
            traindata = get_data(p[1]+"train.json",num=limit)
            train_sets.extend(traindata)
            test_sets.append(get_data(p[1]+"test.json"))
            valid_sets.append(get_data(p[1]+"valid.json",num=limit))
            metrics.append(p[2])

        for key,p in dataformat.new_deepmatcher_data.items():
            traindata = get_data(p[1]+"train.json",num=limit)
            train_sets.extend(traindata)
            test_sets.append(get_data(p[1]+"test.json"))
            valid_sets.append(get_data(p[1]+"valid.json",num=limit))
            metrics.append(p[2])
            
        train_data_loaders = []
        test_data_loaders = []
        valid_data_loaders = []
        
        train_sets = [train_sets]
        if args.model in ['bert','deberta_base','deberta_large','distilbert','mpnet']:
            for i in range(len(train_sets)):
                fea = predata.convert_examples_to_features([ [x[0]+" [SEP] " +x[1]] for x in train_sets[i] ], [int(x[2]) for x in train_sets[i]], args.max_seq_length, tokenizer)
                train_data_loaders.extend(predata.convert_fea_to_tensor(fea, args.batch_size, do_train=1))
            for i in range(len(valid_sets)):
                fea = predata.convert_examples_to_features([ [x[0]+" [SEP] " +x[1]] for x in valid_sets[i] ], [int(x[2]) for x in valid_sets[i]], args.max_seq_length, tokenizer)
                valid_data_loaders.append(predata.convert_fea_to_tensor(fea, args.batch_size, do_train=0))
            for i in range(len(test_sets)):
                fea = predata.convert_examples_to_features([ [x[0]+" [SEP] "+x[1]] for x in test_sets[i] ], [int(x[2]) for x in test_sets[i]], args.max_seq_length, tokenizer)
                test_data_loaders.append(predata.convert_fea_to_tensor(fea, args.batch_size, do_train=0))
        if args.model in ['roberta','distilroberta']:
            for i in range(len(train_sets)):
                fea = predata.convert_examples_to_features_roberta([ [x[0]+" [SEP] "+x[1]] for x in train_sets[i] ], [int(x[2]) for x in train_sets[i]], args.max_seq_length, tokenizer)
                train_data_loaders.extend(predata.convert_fea_to_tensor(fea, args.batch_size, do_train=1))
            for i in range(len(valid_sets)):
                fea = predata.convert_examples_to_features_roberta([ [x[0]+" [SEP] "+x[1]] for x in valid_sets[i] ], [int(x[2]) for x in valid_sets[i]], args.max_seq_length, tokenizer)
                valid_data_loaders.append(predata.convert_fea_to_tensor(fea, args.batch_size, do_train=0))
            for i in range(len(test_sets)):
                fea = predata.convert_examples_to_features_roberta([ [x[0]+" [SEP] "+x[1]] for x in test_sets[i] ], [int(x[2]) for x in test_sets[i]], args.max_seq_length, tokenizer)
                test_data_loaders.append(predata.convert_fea_to_tensor(fea, args.batch_size, do_train=0))
        if args.model=='xlnet':
            for i in range(len(train_sets)):
                fea = predata.convert_examples_to_features([ [x[0]+" [SEP] "+x[1]] for x in train_sets[i] ], [int(x[2]) for x in train_sets[i]], args.max_seq_length, tokenizer, cls_token='<cls>', sep_token='<sep>')
                train_data_loaders.extend(predata.convert_fea_to_tensor(fea, args.batch_size, do_train=1))
            for i in range(len(valid_sets)):
                fea = predata.convert_examples_to_features([ [x[0]+" [SEP] "+x[1]] for x in valid_sets[i] ], [int(x[2]) for x in valid_sets[i]], args.max_seq_length, tokenizer, cls_token='<cls>', sep_token='<sep>')
                valid_data_loaders.append(predata.convert_fea_to_tensor(fea, args.batch_size, do_train=0))
            for i in range(len(test_sets)):
                fea = predata.convert_examples_to_features([ [x[0]+" [SEP] "+x[1]] for x in test_sets[i] ], [int(x[2]) for x in test_sets[i]], args.max_seq_length, tokenizer, cls_token='<cls>', sep_token='<sep>')
                test_data_loaders.append(predata.convert_fea_to_tensor(fea, args.batch_size, do_train=0))
        
        random.seed(args.seed)
        random.shuffle(train_data_loaders)
        print("train datasets num: ",len(train_data_loaders))
        print("test datasets num: ",len(test_data_loaders))
        print("valid datasets num: ",len(valid_data_loaders))
        encoder, moelayer, classifiers = pretrain.train_moe(args, encoder, moelayer, classifiers, [train_data_loaders], valid_data_loaders, metrics)
    
    f1s = []
    recalls = []
    accs = []
    for k in range(len(test_data_loaders)):
        print("test datasets : ",k+1)
        if metrics[k]=='hit': # for EA
            prob = evaluate.evaluate_moe(encoder, moelayer, classifiers, test_data_loaders[k], args=args, flag="get_prob", prob_name="prob.json")
            evaluate.calculate_hits_k(test_sets[k], prob)
        else:
            f1, recall, acc = evaluate.evaluate_moe(encoder, moelayer, classifiers, test_data_loaders[k], args=args, all=1)
            f1s.append(f1)
            recalls.append(recall)
            accs.append(acc)
    print("F1: ", f1s)
    print("Recall: ", recalls)
    print("ACC.", accs)
            

if __name__ == '__main__':
    main()
