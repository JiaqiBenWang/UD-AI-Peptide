import torch
import torch.utils.data as Data
from torch.optim.lr_scheduler import _LRScheduler

src_vocab = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
             'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
             'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}

def make_data(features):
    enc_inputs = []
    for seq in features:
        enc_input = [src_vocab[n] for n in seq]
        enc_inputs.append(enc_input)

    return torch.LongTensor(enc_inputs)

class MyDataSet(Data.Dataset):

    def __init__(self, enc_inputs,labels):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.labels = labels

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.labels[idx]

class LinearWarmUpScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, start_lr, end_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.end_lr = end_lr
        super(LinearWarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr = self.start_lr + (self.end_lr - self.start_lr) * (self.last_epoch / self.warmup_steps)
        else:
            lr = self.end_lr
        return [lr for _ in self.optimizer.param_groups]
