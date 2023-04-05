import torch
import torch.nn as nn

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = x * torch.log(x)
        b = -1.0 * torch.sum(b,dim=1)
        b = torch.mean(b)
        return b

class MoEModule(nn.Module):# if you are not using pytorch lightning, you can also use 'Module'
    def __init__(self, input_size, units, num_experts, num_tasks=None, use_cuda=True, use_expert_bias=False, use_gate_bias=False, expert_activation=None, load_balance=False):
        super(MoEModule, self).__init__()
        self.dropout = nn.Dropout(p=0.05)
        self.expert_kernels = nn.ModuleList([nn.Sequential(
                                nn.Linear(input_size, units),
                                nn.BatchNorm1d(units),
                                nn.LeakyReLU()
                                ) for i in range(num_experts)])

        #one gate
        self.gate_kernel = nn.Sequential(
                            nn.Linear(input_size,units),
                            nn.LeakyReLU(),
                            nn.Linear(units,num_experts),
                            nn.LeakyReLU()
                            )
                            
        self.apply(self.init_bert_weights)
        self.load_balance = load_balance

    def forward(self, x):

        entropy_criterion = EntropyLoss()
        gate_outputs = []
        final_outputs = []
        expert_outputs = []
        for expert_kernel in self.expert_kernels:
            expert_outputs.append(expert_kernel(x))
        expert_outputs = torch.stack(expert_outputs,0)
        expert_outputs = expert_outputs.permute(1,2,0)
        
        gate_output = self.gate_kernel(x)
        cvloss = 0
        entropy_loss = 0
        gate_output = nn.Softmax(dim=-1)(gate_output)
        gate_batch_sum = torch.sum(gate_output,dim=0)
        if self.load_balance:
            
            cvloss = torch.std(gate_batch_sum, unbiased=True) / torch.mean(gate_batch_sum)
            cvloss = torch.pow(cvloss,2)
            entropy_loss = entropy_criterion(gate_output)
        expanded_gate_output = torch.unsqueeze(gate_output, 1)
        weighted_expert_output = expert_outputs * expanded_gate_output.expand_as(expert_outputs)
        final_output = torch.sum(weighted_expert_output, 2)
        
        if self.load_balance:
            return final_output,cvloss,entropy_loss,gate_batch_sum
        else:
            return final_output,gate_batch_sum
        
    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()