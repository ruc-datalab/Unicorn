import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, DistilBertModel, RobertaModel, AutoModel, DebertaModel, XLNetModel
from unicorn.utils import param


class BertEncoder(nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-multilingual-cased')

    def forward(self, x, mask=None, segment=None):
        outputs = self.encoder(x, attention_mask=mask, token_type_ids=segment)
        if param.cls:
            feat = outputs[1]
        else:
            token_len = outputs[0].shape[1]
            feat = torch.sum(outputs[0], dim=1)
            feat = torch.div(feat, token_len)
        return feat

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class MPEncoder(nn.Module):
    def __init__(self):
        super(MPEncoder, self).__init__()
        model_path= "all-mpnet-base-v2"
        self.encoder = AutoModel.from_pretrained(model_path)

    def forward(self, x, mask=None, segment=None):
        inp = {'input_ids':x.detach(), 'attention_mask':mask.detach()}
        outputs = self.encoder(**inp)
        if param.cls:
            feat = outputs[1]
        else:
            feat = mean_pooling(outputs, mask.detach())
            feat = F.normalize(feat, p=2, dim=1)
        return feat



class DistilBertEncoder(nn.Module):
    def __init__(self):
        super(DistilBertEncoder, self).__init__()
        self.encoder = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
        self.pooler = nn.Linear(param.hidden_size, param.hidden_size)

    def forward(self, x, mask=None):
        outputs = self.encoder(x, attention_mask=mask)
        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        feat = self.pooler(pooled_output)
        return feat


class RobertaEncoder(nn.Module):
    def __init__(self):
        super(RobertaEncoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained('roberta-base')

    def forward(self, x, mask=None,segment=None):
        outputs = self.encoder(x, attention_mask=mask)
        sequence_output = outputs[0]
        feat = sequence_output[:, 0, :]
        return feat


class DistilRobertaEncoder(nn.Module):
    def __init__(self):
        super(DistilRobertaEncoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained('distilroberta-base')
        self.pooler = nn.Linear(param.hidden_size, param.hidden_size)

    def forward(self, x, mask=None):
        outputs = self.encoder(x, attention_mask=mask)
        sequence_output = outputs[0]
        feat = sequence_output[:, 0, :]
        return feat

class XLNetEncoder(nn.Module):
    def __init__(self):
        super(XLNetEncoder, self).__init__()
        self.encoder = XLNetModel.from_pretrained("xlnet-base-cased")

    def forward(self, x, mask=None, segment=None):
        outputs = self.encoder(x, attention_mask=mask, token_type_ids=segment)
        if param.cls:
            feat = outputs.last_hidden_state
            feat = feat[:, 0, :]
        else:
            token_len = outputs[0].shape[1]
            feat = torch.sum(outputs[0], dim=1)
            feat = torch.div(feat, token_len)
        return feat

class DebertaBaseEncoder(nn.Module):
    def __init__(self):
        super(DebertaBaseEncoder, self).__init__()
        self.encoder = DebertaModel.from_pretrained("deberta-base")

    def forward(self, x, mask=None, segment=None):
        outputs = self.encoder(x, attention_mask=mask, token_type_ids=segment)
        if param.cls:
            feat = outputs.last_hidden_state
            feat = feat[:, 0, :]
        else:
            token_len = outputs[0].shape[1]
            feat = torch.sum(outputs[0], dim=1)
            feat = torch.div(feat, token_len)
        return feat

class DebertaLargeEncoder(nn.Module):
    def __init__(self):
        super(DebertaLargeEncoder, self).__init__()
        self.encoder = DebertaModel.from_pretrained("deberta-large")

    def forward(self, x, mask=None, segment=None):
        outputs = self.encoder(x, attention_mask=mask, token_type_ids=segment)
        if param.cls:
            feat = outputs.last_hidden_state
            feat = feat[:, 0, :]
        else:
            token_len = outputs[0].shape[1]
            feat = torch.sum(outputs[0], dim=1)
            feat = torch.div(feat, token_len)
        return feat

