from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import torch

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, exm_id, task_id=-1):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.exm_id = exm_id
        self.task_id = task_id


def convert_fea_to_tensor(features_list, batch_size, do_train):
    features = [x[0] for x in features_list]

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_exm_ids = torch.tensor([f.exm_id for f in features], dtype=torch.long)
    all_task_ids = torch.tensor([f.task_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_exm_ids, all_task_ids)
    
    
    if do_train == 0:
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    else:
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=True)
    return dataloader



def convert_examples_to_features(pairs=None, labels=None, max_seq_length=128, tokenizer=None,
                                    cls_token="[CLS]", sep_token='[SEP]',
                                    pad_token=0,task_ids=None):
    features = []
    if labels == None:
        labels = [0] * len(pairs)
    for ex_index, pair in enumerate(pairs):
        if (ex_index + 1) % 200 == 0:
            print("writing example %d of %d" % (ex_index + 1, len(pairs))) 
        if 1:
            fea_pair = []
            for i,tuple in enumerate(pair):  
                if sep_token in tuple:
                    input_ids,input_mask,segment_ids = convert_one_example_to_features_sep(tuple,max_seq_length,cls_token,sep_token,pad_token,tokenizer)                    
                else:
                    input_ids,input_mask,segment_ids = convert_one_example_to_features(tuple,max_seq_length,cls_token,sep_token,pad_token,tokenizer)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                if task_ids:
                    fea_pair.append(
                        InputFeatures(input_ids = input_ids,
                                input_mask = input_mask,
                                segment_ids = segment_ids,
                                label_id = labels[ex_index],
                                exm_id = ex_index,
                                task_id = task_ids[ex_index]) )
                else:
                    fea_pair.append(
                        InputFeatures(input_ids = input_ids,
                                input_mask = input_mask,
                                segment_ids = segment_ids,
                                label_id = labels[ex_index],
                                exm_id = ex_index,
                                task_id = -1) )
        else:
            continue
        features.append(fea_pair)
    return features


def convert_examples_to_features_roberta(pairs=None, labels=None, max_seq_length=128, tokenizer=None,
                                    cls_token="<s>", sep_token='</s>',
                                    pad_token=0):
    features = []
    if labels == None:
        labels = [0] * len(pairs)
    for ex_index, pair in enumerate(pairs):
        if (ex_index + 1) % 200 == 0:
            print("writing example %d of %d" % (ex_index + 1, len(pairs))) 
        if 1:
            fea_pair = []
            for i,tuple in enumerate(pair):  
                if sep_token in tuple:
                    input_ids,input_mask,segment_ids = convert_one_example_to_features_roberta_sep(tuple,max_seq_length,cls_token,sep_token,pad_token,tokenizer)                    
                else:
                    input_ids,input_mask,segment_ids = convert_one_example_to_features(tuple,max_seq_length,cls_token,sep_token,pad_token,tokenizer)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                fea_pair.append(
                    InputFeatures(input_ids = input_ids,
                            input_mask = input_mask,
                            segment_ids = segment_ids,
                            label_id = labels[ex_index],
                            exm_id = ex_index) )
        else:
            continue
        features.append(fea_pair)
    return features

def convert_one_example_to_features(tuple,max_seq_length,cls_token,sep_token,pad_token,tokenizer):    
    tokens = tokenizer.tokenize(tuple)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]
    tokens = [cls_token] + tokens + [sep_token]
    segment_ids = [0]*(len(tokens))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    input_mask = input_mask + ([0] * padding_length)
    segment_ids = segment_ids + ([0] * padding_length)
    
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids,input_mask,segment_ids

def convert_one_example_to_features_sep(tuple,max_seq_length,cls_token,sep_token,pad_token,tokenizer):    
    left = tuple.split(sep_token)[0]
    right = tuple.split(sep_token)[1]
    ltokens = tokenizer.tokenize(left)
    rtokens = tokenizer.tokenize(right)
    more = len(ltokens) + len(rtokens) - max_seq_length + 3
    if more > 0:
        if more <len(rtokens) : 
            rtokens = rtokens[:(len(rtokens) - more)]
        elif more <len(ltokens):
            ltokens = ltokens[:(len(ltokens) - more)]
        else:
            rtokens = rtokens[:50]
            ltokens = ltokens[:50]
    tokens = [cls_token] + ltokens + [sep_token] + rtokens + [sep_token]
    segment_ids = [0]*(len(ltokens)+2) + [1]*(len(rtokens)+1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    input_mask = input_mask + ([0] * padding_length)
    segment_ids = segment_ids + ([0] * padding_length)
    
    return input_ids,input_mask,segment_ids

def convert_one_example_to_features_roberta_sep(tuple,max_seq_length,cls_token,sep_token,pad_token,tokenizer):
    left = tuple.split(sep_token)[0]
    right = tuple.split(sep_token)[1]
    ltokens = tokenizer.tokenize(left)
    rtokens = tokenizer.tokenize(right)
    more = len(ltokens) + len(rtokens) - max_seq_length + 4
    if more > 0:
        if more <len(rtokens) : 
            rtokens = rtokens[:(len(rtokens) - more)]
        elif more <len(ltokens):
            ltokens = ltokens[:(len(ltokens) - more)]
        else:
            rtokens = rtokens[:50]
            ltokens = ltokens[:50]
    tokens = [cls_token] + ltokens + [sep_token] + [sep_token] + rtokens + [sep_token]
    segment_ids = [0]*(len(ltokens)+2) + [1]*(len(rtokens)+1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    input_mask = input_mask + ([0] * padding_length)
    segment_ids = segment_ids + ([0] * padding_length)
    
    return input_ids,input_mask,segment_ids




def convert_fea_to_tensor_one_tuple(features_list, batch_size, do_train):
    features = [x[0] for x in features_list]
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_exm_ids = torch.tensor([f.exm_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_exm_ids)
    sampler = SequentialSampler(dataset)
    if do_train == 0:
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    else:
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=True)
    return dataloader