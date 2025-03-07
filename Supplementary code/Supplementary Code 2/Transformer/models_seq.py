import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=4):  # max_len 设为 4
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 10000^{2i/d_model}
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: [seq_len, batch_size, d_model]
        :return: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).to(torch.bool)
    # print(f"Subsequence mask shape: {subsequence_mask.shape}, dtype: {subsequence_mask.dtype}")
    return subsequence_mask # [batch_size, tgt_len, tgt_len]

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v, d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Q.size(-1))
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        self.args = args
        self.W_Q = nn.Linear(args.d_model, args.d_k * args.n_heads, bias=False)
        self.W_K = nn.Linear(args.d_model, args.d_k * args.n_heads, bias=False)
        self.W_V = nn.Linear(args.d_model, args.d_v * args.n_heads, bias=False)
        self.fc = nn.Linear(args.n_heads * args.d_v, args.d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        :param input_Q: [batch_size, len_q, d_model]
        :param input_K: [batch_size, len_k, d_model]
        :param input_V: [batch_size, len_v, d_model]
        :param attn_mask: [batch_size, seq_len, seq_len]
        :return: [batch_size, len_q, d_model]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.args.n_heads, self.args.d_k).transpose(1,2)  # Q:[batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.args.n_heads, self.args.d_k).transpose(1,2)  # K:[batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.args.n_heads, self.args.d_v).transpose(1,2)  # V:[batch_size, n_heads, len_v, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.args.n_heads, 1, 1)  # [batch_size, n_heads, seq_len, seq_len]

        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.args.n_heads * self.args.d_v)
        output = self.fc(context)

        return nn.LayerNorm(self.args.d_model).to(device)(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args):
        super(PoswiseFeedForwardNet, self).__init__()
        self.args = args
        self.fc = nn.Sequential(
            nn.Linear(args.d_model, args.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(args.d_ff, args.d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        :param inputs: [batch_size, seq_len, d_model]
        :return: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.args.d_model).to(device)(output + residual)


# Encoder 层
class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        :param enc_inputs: [batch_size, src_len, d_model]
        :param enc_self_attn_mask: [batch_size, src_len, src_len]
        :return: [batch_size, src_len, d_model]
        '''

        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self,args):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(args.src_vocab_size, args.d_model)
        self.pos_emb = PositionalEncoding(args.d_model)
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layers)])

    def forward(self, enc_inputs):
        '''
        :param enc_inputs: [batch_size, src_len]
        :return:[batch_size, src_len, d_model]
        '''
        #print("enc_inputs:", enc_inputs)
        #print("Max index in enc_inputs:", enc_inputs.max().item())
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0,1)).transpose(0,1) # [batch_size, src_len, src_len]
        enc_self_attn_mask = get_attn_subsequence_mask(enc_inputs) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Classifier(nn.Module):
    def __init__(self, input_dim, args):
        super().__init__()

        self.args = args
        last_layer = 1  # For regression, output dimension is 1

        hidden_layer = [512, 256, 64, 32, last_layer]

        self.sigmoid = nn.Sigmoid()

        self.e1 = nn.Linear(input_dim, hidden_layer[0])
        self.e2 = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.e3 = nn.Linear(hidden_layer[1], hidden_layer[2])
        self.e4 = nn.Linear(hidden_layer[2], hidden_layer[3])
        self.e5 = nn.Linear(hidden_layer[3], hidden_layer[4])

        self.dropout = nn.Dropout(p=0.3)
        self.batchnorm0 = nn.BatchNorm1d(hidden_layer[0])
        self.batchnorm1 = nn.BatchNorm1d(hidden_layer[1])
        self.batchnorm2 = nn.BatchNorm1d(hidden_layer[2])
        self.batchnorm3 = nn.BatchNorm1d(hidden_layer[3])

    def forward(self, dec_input):
        h_1 = F.leaky_relu(self.batchnorm0(self.e1(dec_input)), negative_slope=0.05, inplace=True)
        h_1 = self.dropout(h_1)
        h_2 = F.leaky_relu(self.batchnorm1(self.e2(h_1)), negative_slope=0.05, inplace=True)
        h_2 = self.dropout(h_2)
        h_3 = F.leaky_relu(self.e3(h_2), negative_slope=0.1, inplace=True)
        h_3 = self.dropout(h_3)
        h_4 = F.leaky_relu(self.e4(h_3), negative_slope=0.1, inplace=True)
        y = self.e5(h_4)

        # For regression
        return self.sigmoid(y) * (self.args.max - self.args.min) + self.args.min


class Transformer(nn.Module):
    def __init__(self,args):
        super(Transformer,self).__init__()
        self.encoder = Encoder(args).to(device)
        self.decoder = Classifier(args.src_len*args.d_model,args).to(device)

    def forward(self,enc_inputs):
        '''
        :param enc_inputs: [batch_size, src_len]
        :param dec_inputs: [batch_size, tgt_len]
        :return:
        '''
        enc_outputs,_ = self.encoder(enc_inputs)
        dec_inputs = torch.reshape(enc_outputs,(enc_outputs.shape[0],-1))
        pred = self.decoder(dec_inputs)

        return pred
