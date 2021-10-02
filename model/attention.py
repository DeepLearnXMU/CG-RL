# coding=utf-8

import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, q_size, k_size, hidden_size, name="attention"):
        super(Attention, self).__init__(name)

        self._q_size = q_size
        self._k_size = k_size
        self._hidden_size = hidden_size

        self.q_transform = nn.Linear(q_size, hidden_size)
        self.k_transform = nn.Linear(k_size, hidden_size)
        self.v_transform = nn.Linear(hidden_size, 1)

    def compute_cache(self, memory):
        return self.k_transform(memory)

    def forward(self, query, memory):
        q = self.q_transform(query)

        k = self.k_transform(memory)


        # q: [batch, 1, hidden_size]
        # k: [batch, length, hidden_size]
        logits = self.v_transform(torch.tanh(q + k))
        # [batch, length, 1]
        logits = torch.transpose(logits, 1, 2)
        # [batch, 1, 1, length]
        logits = torch.unsqueeze(logits, 2)

        weights = torch.softmax(logits, dim=-1)

        # [batch, 1, length]
        weights = torch.squeeze(weights, 2)
        output = torch.matmul(weights, memory)

        return output


class MultiHeadAttentionBase(nn.Module):

    def __init__(self):
        super(MultiHeadAttentionBase, self).__init__()

    @staticmethod
    def split_heads(x, heads):
        batch = x.shape[0]
        length = x.shape[1]
        channels = x.shape[2]

        y = torch.reshape(x, [batch, length, heads, channels // heads])
        return torch.transpose(y, 2, 1)

    @staticmethod
    def combine_heads(x):
        batch = x.shape[0]
        heads = x.shape[1]
        length = x.shape[2]
        channels = x.shape[3]

        y = torch.transpose(x, 2, 1)

        return torch.reshape(y, [batch, length, heads * channels])
    
    @staticmethod
    def masking_bias(mask, inf=-1e9):
        ret = (1.0 - mask) * inf
        return torch.unsqueeze(torch.unsqueeze(ret, 1), 1)


class MultiHeadAttention(MultiHeadAttentionBase):

    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.q_transform = nn.Linear(hidden_size, hidden_size)
        self.k_transform = nn.Linear(hidden_size, hidden_size)
        self.v_transform = nn.Linear(hidden_size, hidden_size)
        self.o_transform = nn.Linear(hidden_size, hidden_size)


    def forward(self, query, masks, memory=None):
        bias = self.masking_bias(masks)
        q = self.q_transform(query)

        if memory is not None:
            k, v = None, None

            # encoder-decoder attention
            k = k or self.k_transform(memory)
            v = v or self.v_transform(memory)
        else:
            # self-attention
            k = self.k_transform(query)
            v = self.v_transform(query)


        # split heads (heads, seq_len, channels // heads)
        qh = self.split_heads(q, self.num_heads)
        kh = self.split_heads(k, self.num_heads)
        vh = self.split_heads(v, self.num_heads)

        # scale query
        qh = qh * (self.hidden_size // self.num_heads) ** -0.5

        # dot-product attention
        kh = torch.transpose(kh, -2, -1)
        logits = torch.matmul(qh, kh)
        if bias is not None:
            logits = logits + bias
        weights = torch.nn.functional.dropout(torch.softmax(logits, dim=-1),
                                              p=self.dropout,
                                              training=self.training)
        x = torch.matmul(weights, vh)
        # combine heads
        output = self.o_transform(self.combine_heads(x))

        return output

